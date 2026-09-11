"""供调度员复核的结构化调度报告。

每次重排后生成一份 :class:`ReplanSnapshot`（不覆盖历史，便于比较
"临时换机 / 耗电突增前后"的计划差异）；打烊时汇总为 :class:`DispatchReport`，
统一包含：

- 场景时间：仿真绝对分钟、钟点、相对当前分钟、营业高峰；
- 每支麦克风的预计断电（TTE，绝对 + 相对）与关联的预警 / 换电动作；
- 换电计划：原因、备电阈值、计划时刻（绝对分钟 + 钟点 + 距当前）；
- 未解决的备电不足（计划里没有换电条目的告警 / 校验后仍断电）；
- 最终断电统计（开班已断电、营业中断电、各次重排快照留存）。

时间表示一律双轨：``absolute`` 是仿真绝对分钟（可换算钟点），
``relative`` 是"距该快照生成时刻还有多少分钟"，二者不混用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .battery import time_to_empty, CUTOFF_SOC
from .engine import clone_scene, simulate
from .models import Scene, PlannedSwap
from .monitor import Alert

# 重排触发来源（标签会原样保留在快照里）
TRIGGER_INITIAL = "开班初始计划"
TRIGGER_AD_HOC_SWAP = "临时换机后重排"
TRIGGER_DEVICE_REPLACE = "更换机身后重排"
TRIGGER_DRAIN_SPIKE = "耗电突增后重排"
TRIGGER_OTHER = "重排"

# 视为"未解决备电不足"的提示语关键词（来自 planner 兜底失败分支）
_SHORTAGE_KEYWORDS = ("无满电备电", "备电严重不足", "无达标备电",
                      "截止电压", "强制")


@dataclass
class TimeRef:
    """时间双轨：仿真绝对分钟 / 钟点 / 距快照生成时刻的相对分钟。"""
    absolute: int
    clock: str
    relative: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanEntryView:
    time: TimeRef
    mic_id: str
    reason: str
    min_spare_soc: float
    forced: bool
    tte_at_swap: float | None      # 不换电时该时刻的预计断电剩余分钟（可空）
    related_alert: str | None = None   # 关联的最近一条该麦预警文本


@dataclass
class MicForecast:
    mic_id: str
    soc_now: float
    tte_absolute: int | None       # 预计断电的仿真绝对分钟
    tte_clock: str | None
    tte_relative: int | None       # 距快照生成时刻的分钟数（0 = 已断电）
    initial_dead: bool             # 快照生成时是否已低于截止电压
    related_alert: str | None = None
    related_swap: TimeRef | None = None  # 为它安排的首次换电时刻


@dataclass
class UnresolvedShortage:
    mic_id: str
    advisory: str
    initial_dead: bool
    outage: bool                   # 完整校验后是否确实发生断电
    outage_time: TimeRef | None


@dataclass
class OutageRecord:
    time: TimeRef
    mic_id: str
    initial: bool                  # True=开班已断电；False=营业中跌破


@dataclass
class PlanDiff:
    """相邻两次重排的计划差异（绝对分钟对齐，避免 now 漂移导致误判）。"""
    added: list[PlanEntryView] = field(default_factory=list)
    removed: list[PlanEntryView] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)  # {mic_id, old, new}

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


@dataclass
class ReplanSnapshot:
    """一次重排的完整留存（含失败 / 无备电场景，绝不丢弃）。"""
    seq: int
    trigger: str
    note: str
    generated_at: TimeRef
    horizon: TimeRef
    peak_bands: list[tuple[str, str]]
    plan: list[PlanEntryView]
    forecasts: list[MicForecast]
    advisories: list[str]
    unresolved: list[UnresolvedShortage]
    validated_outages: int         # 用该计划完整快放校验后的断电条数
    validated_failed: int          # 无法执行的换机条数
    alerts: list[dict] = field(default_factory=list)


@dataclass
class DispatchReport:
    scene_start: TimeRef
    horizon: TimeRef
    peak_bands: list[tuple[str, str]]
    finalized_at: TimeRef
    snapshots: list[ReplanSnapshot]
    outages: list[OutageRecord]
    alerts: list[dict]

    # ---- 导出 ----
    def to_dict(self) -> dict:
        def emit(obj):
            if isinstance(obj, TimeRef):
                return obj.as_dict()
            if isinstance(obj, (PlanEntryView, MicForecast,
                                UnresolvedShortage, OutageRecord,
                                ReplanSnapshot)):
                return {k: emit(v) for k, v in asdict(obj).items()}
            if isinstance(obj, PlanDiff):
                return {k: emit(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [emit(v) for v in obj]
            if isinstance(obj, tuple):
                return [emit(v) for v in obj]
            return obj
        return {
            "scene_start": emit(self.scene_start),
            "horizon": emit(self.horizon),
            "peak_bands": [list(b) for b in self.peak_bands],
            "finalized_at": emit(self.finalized_at),
            "snapshots": [emit(s) for s in self.snapshots],
            "outages": [emit(o) for o in self.outages],
            "alerts": self.alerts,
        }

    def to_json(self, path: str | None = None,
                indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text


def clock_of(minute: int, day_start_hour: int = 0) -> str:
    """仿真绝对分钟 -> HH:MM（跨零点自动取模 24h）。"""
    t = (minute + day_start_hour * 60) % (24 * 60)
    return f"{t // 60:02d}:{t % 60:02d}"


class DispatchReporter:
    """累积重排快照与现场事件，最后产出 / 导出调度报告。"""

    def __init__(self, scene: Scene, day_start_hour: int = 0):
        self.day_start_hour = day_start_hour
        self._snapshots: list[ReplanSnapshot] = []
        self._alerts: list[Alert] = []
        self._outage_keys: set[tuple[int, str]] = set()
        self._outages: list[OutageRecord] = []
        # 开班时已经低于截止电压的麦（用于断电分类）
        self._initial_dead = {m.mic_id for m in scene.mics
                              if m.battery.soc <= CUTOFF_SOC}

    # ---- 时间工具 ----
    def _tref(self, minute: int, now: int) -> TimeRef:
        return TimeRef(minute, clock_of(minute, self.day_start_hour),
                       minute - now)

    def _band_clocks(self, scene: Scene) -> list[tuple[str, str]]:
        return [(clock_of(a, self.day_start_hour),
                 clock_of(b, self.day_start_hour))
                for a, b in scene.peak_bands]

    # ---- 事件登记 ----
    def record_alert(self, alert: Alert) -> None:
        self._alerts.append(alert)

    def record_outage(self, minute: int, mic_id: str,
                      now: int) -> OutageRecord:
        """登记一次实际断电；initial=True 表示该麦开班时就已低于截止电压。"""
        key = (minute, mic_id)
        if key in self._outage_keys:
            return next(o for o in self._outages
                        if (o.time.absolute, o.mic_id) == key)
        rec = OutageRecord(self._tref(minute, now), mic_id,
                           initial=mic_id in self._initial_dead
                           and minute <= now)
        self._outage_keys.add(key)
        self._outages.append(rec)
        return rec

    # ---- 重排快照 ----
    def record_replan(self, scene: Scene, plan: list[PlannedSwap],
                      advisories: list[str], trigger: str,
                      note: str = "") -> ReplanSnapshot:
        """根据当前现场 + 计划生成一份快照并留存（含无备电/失败场景）。"""
        now = scene.now
        seq = len(self._snapshots) + 1

        # 最近一次各麦预警文本（关联换电动作用）
        latest_alert: dict[str, str] = {}
        for a in self._alerts:
            if a.mic_id:
                latest_alert[a.mic_id] = a.text

        # 为每支麦安排的首次换电时刻
        first_swap: dict[str, PlannedSwap] = {}
        for s in sorted(plan, key=lambda x: x.time):
            first_swap.setdefault(s.mic_id, s)

        # 各麦 TTE 预测（绝对分钟 + 相对分钟双轨）
        forecasts: list[MicForecast] = []
        for m in scene.mics:
            tte = time_to_empty(m, now, scene.horizon)
            if tte is None:
                abs_t = clock = rel = None
            else:
                abs_t = int(round(tte))
                clock = clock_of(abs_t, self.day_start_hour)
                rel = max(0, abs_t - now)
            initial_dead = m.battery.soc <= CUTOFF_SOC
            fs = first_swap.get(m.mic_id)
            forecasts.append(MicForecast(
                mic_id=m.mic_id,
                soc_now=m.battery.soc,
                tte_absolute=abs_t,
                tte_clock=clock,
                tte_relative=rel,
                initial_dead=initial_dead,
                related_alert=latest_alert.get(m.mic_id),
                related_swap=(self._tref(fs.time, now) if fs else None),
            ))

        # 计划条目视图
        entries: list[PlanEntryView] = []
        for s in sorted(plan, key=lambda x: (x.time, x.mic_id)):
            tte_at = time_to_empty(scene.mic(s.mic_id), s.time,
                                   scene.horizon)
            entries.append(PlanEntryView(
                time=self._tref(s.time, now),
                mic_id=s.mic_id,
                reason=s.reason,
                min_spare_soc=s.min_spare_soc,
                forced=s.forced,
                tte_at_swap=(round(s.time - tte_at, 1)
                             if tte_at is not None else None),
                related_alert=latest_alert.get(s.mic_id),
            ))

        # 计划完整快放校验：哪些麦确实断电、哪些换机无法执行
        sim_scene = clone_scene(scene)
        result = simulate(sim_scene, plan)
        planned_mics = {s.mic_id for s in plan}
        outage_map = {mic: t for t, mic in result.outages}

        # 未解决的备电不足（两类，都以"完整快放后是否真的断电"为准，
        # 半电应急 / 成功的强制换机虽带"无满电备电"字样，但风险已化解，
        # 不算未解决）：
        #  a) planner 给出的备电紧张类提示，且该麦校验中确实断电；
        #  b) 校验后断电但计划里完全没有该麦换电条目的。
        unresolved: list[UnresolvedShortage] = []
        seen_mics: set[str] = set()
        # 快照生成时刻已低于截止电压的麦（区分"初始已断电"与"中途断电"）
        dead_now = {m.mic_id for m in scene.mics
                    if m.battery.soc <= CUTOFF_SOC}
        for text in advisories:
            mid = _mic_in_text(text)
            if not mid or not any(k in text for k in _SHORTAGE_KEYWORDS):
                continue
            outage_t = outage_map.get(mid)
            if outage_t is None:
                continue  # 已被计划兜住（半电应急 / 强制换机成功）
            seen_mics.add(mid)
            unresolved.append(UnresolvedShortage(
                mic_id=mid,
                advisory=text,
                initial_dead=mid in dead_now,
                outage=True,
                outage_time=self._tref(outage_t, now),
            ))
        for mid, t in outage_map.items():
            if mid in seen_mics or mid in planned_mics:
                continue
            unresolved.append(UnresolvedShortage(
                mic_id=mid,
                advisory=f"{mid} 无换电安排，校验中于 {clock_of(t, self.day_start_hour)} 断电",
                initial_dead=mid in dead_now,
                outage=True,
                outage_time=self._tref(t, now),
            ))

        snap = ReplanSnapshot(
            seq=seq,
            trigger=trigger,
            note=note,
            generated_at=self._tref(now, now),
            horizon=self._tref(scene.horizon, now),
            peak_bands=self._band_clocks(scene),
            plan=entries,
            forecasts=forecasts,
            advisories=list(advisories),
            unresolved=unresolved,
            validated_outages=len(result.outages),
            validated_failed=len(result.failed),
            alerts=[{
                "minute": a.minute,
                "clock": clock_of(a.minute, self.day_start_hour),
                "relative": a.minute - now,
                "level": a.level,
                "mic_id": a.mic_id,
                "source": a.source,
                "text": a.text,
            } for a in self._alerts],
        )
        self._snapshots.append(snap)
        return snap

    def diff_latest(self) -> PlanDiff | None:
        """比较最近两次重排（用于临时换机/耗电突增前后差异）。"""
        if len(self._snapshots) < 2:
            return None
        return self.diff(self._snapshots[-2], self._snapshots[-1])

    @staticmethod
    def diff(old: ReplanSnapshot, new: ReplanSnapshot) -> PlanDiff:
        """按 (mic_id, 绝对分钟) 对齐两次计划。"""
        def key(e: PlanEntryView):
            return (e.mic_id, e.time.absolute)
        old_map = {key(e): e for e in old.plan}
        new_map = {key(e): e for e in new.plan}
        added = [e for k, e in new_map.items() if k not in old_map]
        removed = [e for k, e in old_map.items() if k not in new_map]

        # 同一支麦的首次换机时刻发生位移（成对匹配 added/removed）
        changed: list[dict] = []
        used_add: set[tuple] = set()
        used_rem: set[tuple] = set()
        mics = {e.mic_id for e in added} | {e.mic_id for e in removed}
        for mid in sorted(mics):
            a = sorted((e for e in added if e.mic_id == mid),
                       key=lambda e: e.time.absolute)
            r = sorted((e for e in removed if e.mic_id == mid),
                       key=lambda e: e.time.absolute)
            if a and r:
                ae, re_ = a[0], r[0]
                used_add.add(key(ae))
                used_rem.add(key(re_))
                changed.append({
                    "mic_id": mid,
                    "old": re_.time.as_dict(),
                    "new": ae.time.as_dict(),
                    "shift_minutes": ae.time.absolute - re_.time.absolute,
                })
        added = [e for e in added if key(e) not in used_add]
        removed = [e for e in removed if key(e) not in used_rem]
        return PlanDiff(added=added, removed=removed, changed=changed)

    # ---- 终稿 ----
    def finalize(self, scene: Scene) -> DispatchReport:
        now = scene.now
        return DispatchReport(
            scene_start=self._tref(self._snapshots[0].generated_at.absolute,
                                   now) if self._snapshots
            else self._tref(now, now),
            horizon=self._tref(scene.horizon, now),
            peak_bands=self._band_clocks(scene),
            finalized_at=self._tref(now, now),
            snapshots=list(self._snapshots),
            outages=list(self._outages),
            alerts=[{
                "minute": a.minute,
                "clock": clock_of(a.minute, self.day_start_hour),
                "level": a.level,
                "mic_id": a.mic_id,
                "source": a.source,
                "text": a.text,
            } for a in self._alerts],
        )


def _mic_in_text(text: str) -> str | None:
    """从 planner 提示语里提取麦编号（M1 / M6' 等）。"""
    import re
    m = re.search(r"M[\w']+", text)
    return m.group(0) if m else None


# ---------------------------------------------------------------- 文本渲染

def render_snapshot(snap: ReplanSnapshot) -> str:
    """渲染单次重排快照（每次重排后给调度员即时复核）。"""
    lines = [
        f"── 重排快照 #{snap.seq} [{snap.trigger}]"
        + (f" {snap.note}" if snap.note else ""),
        f"   生成时刻 {snap.generated_at.clock}"
        f"（仿真第 {snap.generated_at.absolute} 分钟），"
        f"打烊 {snap.horizon.clock}（还剩 {snap.horizon.relative} 分钟）",
    ]
    if snap.peak_bands:
        bands = "、".join(f"{a}-{b}" for a, b in snap.peak_bands)
        lines.append(f"   高峰时段: {bands}")

    lines.append("   预计断电（相对当前分钟）:")
    for f in sorted(snap.forecasts,
                    key=lambda x: (x.tte_relative is None,
                                   x.tte_relative if x.tte_relative is not None else 0)):
        if f.tte_relative is None:
            tte_txt = "打烊前不断电"
        elif f.tte_relative == 0 and f.initial_dead:
            tte_txt = "开班已低于截止电压(立即处理)"
        else:
            tte_txt = f"{f.tte_clock} 断电（{f.tte_relative} 分钟后）"
        swap_txt = (f" -> 首次换电 {f.related_swap.clock}"
                    f"（{f.related_swap.relative} 分钟后）"
                    if f.related_swap else " -> 无换电安排")
        lines.append(f"     {f.mic_id:>4} 当前 {f.soc_now:4.0%}  {tte_txt}{swap_txt}")

    if snap.plan:
        lines.append("   换电计划:")
        for e in snap.plan:
            force = " [强制]" if e.forced else ""
            rel = f"第 {e.time.relative} 分钟后" if e.time.relative > 0 else "立即"
            lines.append(
                f"     {e.time.clock}（绝对 {e.time.absolute}，{rel}）"
                f" {e.mic_id}  {e.reason}  备电≥{e.min_spare_soc:.0%}{force}")
            if e.related_alert:
                lines.append(f"       └ 关联预警: {e.related_alert}")
    else:
        lines.append("   换电计划: 无")

    if snap.advisories:
        lines.append("   调度提示:")
        lines.extend(f"     · {a}" for a in snap.advisories)
    if snap.unresolved:
        lines.append("   ⚠ 未解决备电不足:")
        for u in snap.unresolved:
            tag = "开班已断电" if u.initial_dead else "校验中断电"
            state = f"实际 {u.outage_time.clock} 断电（{tag}）" \
                if u.outage and u.outage_time else "未实际断电"
            lines.append(f"     · {u.mic_id}: {u.advisory} [{state}]")
    lines.append(f"   计划快放校验: 断电 {snap.validated_outages} 起，"
                 f"换机落空 {snap.validated_failed} 起")
    return "\n".join(lines)


def render_report(report: DispatchReport) -> str:
    """渲染终稿报告（打烊复核 / 归档）。"""
    lines = [
        "================ 调度报告（终稿）================",
        f"场景起点 {report.scene_start.clock}"
        f"（仿真第 {report.scene_start.absolute} 分钟）  "
        f"打烊 {report.horizon.clock}  "
        f"报告时刻 {report.finalized_at.clock}",
    ]
    if report.peak_bands:
        bands = "、".join(f"{a}-{b}" for a, b in report.peak_bands)
        lines.append(f"高峰时段: {bands}")

    lines.append(f"\n重排记录（共 {len(report.snapshots)} 次，全部留存）:")
    for snap in report.snapshots:
        lines.append(
            f"  #{snap.seq} {snap.generated_at.clock} [{snap.trigger}]"
            f" 计划 {len(snap.plan)} 次 / 未解决 {len(snap.unresolved)} 起"
            f" / 校验断电 {snap.validated_outages} 起"
            + (f" — {snap.note}" if snap.note else ""))

    # 相邻快照差异
    for i in range(1, len(report.snapshots)):
        d = DispatchReporter.diff(report.snapshots[i - 1], report.snapshots[i])
        if d.is_empty():
            continue
        lines.append(f"\n  #{i} -> #{i + 1} 计划差异"
                     f"（{report.snapshots[i - 1].trigger} → "
                     f"{report.snapshots[i].trigger}）:")
        for c in d.changed:
            sign = "+" if c["shift_minutes"] > 0 else ""
            lines.append(
                f"    {c['mic_id']} 换电时刻 {c['old']['clock']} -> "
                f"{c['new']['clock']}（{sign}{c['shift_minutes']} 分钟，"
                f"绝对 {c['old']['absolute']} -> {c['new']['absolute']}）")
        for e in d.added:
            lines.append(f"    + {e.time.clock} {e.mic_id} {e.reason}"
                         f"（新增，{e.time.relative} 分钟后）")
        for e in d.removed:
            lines.append(f"    - {e.time.clock} {e.mic_id} {e.reason}（取消）")

    # 未解决告警汇总（跨所有快照）
    all_unresolved = [(s, u) for s in report.snapshots
                      for u in s.unresolved]
    lines.append("\n未解决的备电不足告警:")
    if all_unresolved:
        for s, u in all_unresolved:
            lines.append(f"  [快照#{s.seq} {s.trigger}] {u.mic_id}: {u.advisory}")
    else:
        lines.append("  ✓ 无")

    lines.append("\n最终断电统计:")
    if report.outages:
        init_n = sum(1 for o in report.outages if o.initial)
        lines.append(f"  合计 {len(report.outages)} 起"
                     f"（开班已断电 {init_n} 起，营业中断电 "
                     f"{len(report.outages) - init_n} 起）:")
        for o in report.outages:
            tag = "开班已断电" if o.initial else "营业中断电"
            lines.append(f"    {o.time.clock}（绝对 {o.time.absolute}，"
                         f"相对报告 {o.time.relative:+d} 分钟）"
                         f" {o.mic_id}  [{tag}]")
    else:
        lines.append("  ✓ 全场无断电（含开班即低电均已处置）")

    lines.append(f"\n预警记录（共 {len(report.alerts)} 条）:")
    for a in report.alerts:
        scope = a["mic_id"] or "充电池"
        lines.append(f"  {a['clock']} [{a['level']}] {scope}: {a['text']}")
    lines.append("================================================")
    return "\n".join(lines)
