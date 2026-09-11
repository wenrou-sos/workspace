"""供调度员复核的结构化调度报告。

每次重排后生成一份 :class:`ReplanSnapshot`（不覆盖历史，便于比较
"临时换机 / 耗电突增前后"的计划差异）；打烊时汇总为 :class:`DispatchReport`，
统一包含：

- 场景时间：仿真绝对分钟、钟点、相对当前分钟、营业高峰；
- 每支麦克风的预计断电（TTE，绝对 + 相对）与关联的预警 / 换电动作；
- 换电计划：原因、备电阈值、计划时刻（绝对分钟 + 钟点 + 距当前）；
- **实际换电台账**（:class:`SwapEvent`）：计划内/临时 × 成功/落空、
  实际换上的电池 UID、落空发生在哪一分钟；终稿按 (绝对分钟, 麦) 把每条
  计划与真实执行对账（已执行 / 落空 / 被取消 / 临时换机后失效 / 未执行）；
- **计划取消台账**（:class:`CancelledPlan`）：按重排原因区分临时换机致旧
  计划失效、换机身迁移、耗电突增提前、普通重排；
- 未解决的备电不足（计划里没有换电条目的告警 / 校验后仍断电）；
- 最终断电统计（开班已断电、营业中断电、各次重排快照留存）。

注意区分两类"落空"：``validated_failed`` 是**影子仿真**对候选计划的预判，
``SwapEvent(kind=planned, result=failed)`` 是主循环里**真实发生**的执行失败
（如同一分钟多支麦抢占同一块备电）。

时间表示一律双轨：``absolute`` 是仿真绝对分钟（可换算钟点），
``relative`` 是"距该快照生成时刻还有多少分钟"，二者不混用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass

from .battery import time_to_empty, predict_soc, CUTOFF_SOC
from .engine import clone_scene, simulate
from .models import Scene, PlannedSwap
from .monitor import Alert

# 重排触发来源（标签会原样保留在快照里）
TRIGGER_INITIAL = "开班初始计划"
TRIGGER_AD_HOC_SWAP = "临时换机后重排"
TRIGGER_DEVICE_REPLACE = "更换机身后重排"
TRIGGER_DRAIN_SPIKE = "耗电突增后重排"
TRIGGER_OTHER = "重排"

# 真实换电动作的类型 / 结果
KIND_PLANNED = "planned"   # 计划内换电
KIND_AD_HOC = "ad_hoc"     # 临时换机（客人要求等，不在计划条目内）
RESULT_SUCCESS = "success"
RESULT_FAILED = "failed"

# 计划条目终稿对账状态
ST_EXECUTED = "executed"        # 已按计划执行
ST_FAILED = "failed"            # 到点执行但落空（如同分钟抢占）
ST_CANCELLED = "cancelled"      # 被后续重排取消（旧计划失效）
ST_NOT_EXECUTED = "not_executed"  # 计划时刻已过，却没有任何执行记录
ST_PENDING = "pending"          # 计划时刻尚未到

# 计划取消原因码
CAUSE_AD_HOC = "ad_hoc"
CAUSE_DEVICE_REPLACE = "device_replace"
CAUSE_DRAIN_SPIKE = "drain_spike"
CAUSE_REPLAN = "replan"

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
    tte_at_swap: float | None      # 若不换电，到换电时刻还剩多少分钟断电
                                   # （非负：绝对断电时刻 - 换电时刻）；
                                   # 0 = 换电时已在截止电压；None = 打烊前不断电
    related_alert: str | None = None   # 关联的最近一条该麦预警文本
    # —— 终稿按绝对分钟与实际执行对账后回填（快照生成时均为待执行）——
    execution: str | None = None       # executed / failed / cancelled / pending / not_executed
    execution_label: str = ""          # 可读取的执行状态文案
    battery_uid: str | None = None     # 实际换上的电池 UID（成功时）
    execution_detail: str = ""         # 落空原因等
    cancel_cause: str = ""              # cancelled 时的机器可读原因码


@dataclass
class SwapEvent:
    """一次真实发生的换电动作（区别于影子仿真校验）。"""
    minute: int                    # 仿真绝对分钟（计划-执行据此关联）
    clock: str
    offset_from_start: int         # 距报告起点的分钟数
    mic_id: str
    kind: str                      # planned（计划内）/ ad_hoc（临时换机）
    result: str                    # success / failed
    battery_uid: str | None        # 实际换上的电池 UID；落空时为 None
    threshold: float               # 当时要求的最低备电电量
    reason: str                    # 计划原因或临时事件说明
    detail: str                    # 落空原因等补充
    plan_seq: int | None           # 计划内换电所属的重排快照序号


@dataclass
class CancelledPlan:
    """一条因重排而作废、且从未执行的旧计划。"""
    mic_id: str
    planned_minute: int
    clock: str
    reason: str
    removed_in_snapshot: int       # 在哪一次重排中被移除
    cause_code: str                # ad_hoc / device_replace / drain_spike / replan
    cause: str                     # 可读原因


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
    swap_events: list[SwapEvent] = field(default_factory=list)
    cancelled_plans: list[CancelledPlan] = field(default_factory=list)

    # ---- 导出 ----
    def to_dict(self) -> dict:
        def emit(obj):
            if is_dataclass(obj) and not isinstance(obj, type):
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
            # 实际换电台账：计划内/临时、成功/落空、实际电池 UID
            "swap_events": [emit(e) for e in self.swap_events],
            # 计划取消台账：旧计划为何失效
            "cancelled_plans": [emit(c) for c in self.cancelled_plans],
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
        # 真实换电台账（主循环执行，不是影子仿真）
        self._swap_events: list[SwapEvent] = []
        # 因重排而作废的旧计划条目
        self._cancelled: list[CancelledPlan] = []
        # 报告起点（第一条快照的生成时刻），台账时间相对它显示
        self._start_minute: int | None = None
        # 开班时已经低于截止电压的麦（用于断电分类）
        self._initial_dead = {m.mic_id for m in scene.mics
                              if m.battery.soc <= CUTOFF_SOC}
        # 已经被成功换电救起过的麦：获救之后的任何再次断电都属于
        # "营业中断电"，不能再归类为开班初始故障
        self._ever_rescued: set[str] = set()

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

    def mark_rescued(self, mic_id: str) -> None:
        """登记该麦已成功换入可开机的电池（获救）。

        获救后再次跌破截止电压属于"营业中断电"，不再算开班初始故障。
        由执行侧在每次成功换电（含临时换机）后调用。
        """
        self._ever_rescued.add(mic_id)

    def record_swap_execution(self, minute: int, mic_id: str, kind: str,
                              result: str, threshold: float, reason: str,
                              battery_uid: str | None = None,
                              detail: str = "") -> SwapEvent:
        """登记一次【真实执行】的换电（区别于影子仿真的 validated_*）。

        - kind=planned：计划内换机，自动关联当时生效的重排快照；
        - kind=ad_hoc：临时换机（不对应任何计划条目，旧计划因此失效）；
        - result=success 时自动 mark_rescued；failed 时 battery_uid=None，
          detail 记录落空原因（如"同分钟无达标备电/被抢占"）。
        """
        if self._start_minute is None:
            self._start_minute = minute
        event = SwapEvent(
            minute=minute,
            clock=clock_of(minute, self.day_start_hour),
            offset_from_start=minute - self._start_minute,
            mic_id=mic_id,
            kind=kind,
            result=result,
            battery_uid=battery_uid,
            threshold=threshold,
            reason=reason,
            detail=detail,
            plan_seq=(self._snapshots[-1].seq
                      if kind == KIND_PLANNED and self._snapshots else None),
        )
        if result == RESULT_SUCCESS:
            self.mark_rescued(mic_id)
        self._swap_events.append(event)
        return event

    def record_outage(self, minute: int, mic_id: str,
                      now: int | None = None) -> OutageRecord:
        """登记一次实际断电。

        分类规则（按实际发生阶段，而非只看编号）：
        - 开班即低于截止电压、且从未被救起 -> ``initial=True``；
        - 曾成功换电后再次跌破 / 运营中自然耗尽 -> ``initial=False``。
        ``now`` 仅用于相对时间展示，不参与初始判定（此前"事件分钟同时
        作为当前时间"导致任何后续断电都满足初始条件）。
        """
        key = (minute, mic_id)
        if key in self._outage_keys:
            return next(o for o in self._outages
                        if (o.time.absolute, o.mic_id) == key)
        is_initial = (mic_id in self._initial_dead
                      and mic_id not in self._ever_rescued)
        rec = OutageRecord(self._tref(minute, now if now is not None else minute),
                           mic_id, initial=is_initial)
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
        if self._start_minute is None:
            self._start_minute = now

        # 与上一版计划按 (麦, 绝对分钟) 对齐：新版不再包含、且当时尚未执行
        # 的旧条目 = 因本次重排而取消（临时换机致旧计划失效 / 换机身迁移 /
        # 耗电突增提前 / 普通重排）。
        if self._snapshots:
            self._detect_cancellations(self._snapshots[-1], plan, seq,
                                       trigger, now)

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
            # 剩余 = 绝对断电时刻 - 换电时刻，恒为非负。先推演到换电时刻
            # 的真实电量，再算 TTE——直接用当前电量会漏掉 now->换电时刻
            # 的耗电、高估续航。起始已低于截止电压 -> 剩余 0；打烊前不断
            # 电 -> None。
            target = scene.mic(s.mic_id)
            soc_at_swap = predict_soc(target, now, s.time)
            tte_at = time_to_empty(target, s.time, scene.horizon,
                                   soc=soc_at_swap)
            remaining = (round(max(0.0, tte_at - s.time), 1)
                         if tte_at is not None else None)
            entries.append(PlanEntryView(
                time=self._tref(s.time, now),
                mic_id=s.mic_id,
                reason=s.reason,
                min_spare_soc=s.min_spare_soc,
                forced=s.forced,
                tte_at_swap=remaining,
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

    # ---- 计划取消检测 ----
    def _detect_cancellations(self, old_snap: ReplanSnapshot,
                              new_plan: list[PlannedSwap], new_seq: int,
                              trigger: str, now: int) -> None:
        """新版计划里消失的旧条目 -> 取消台账（区分取消原因）。"""
        new_keys = {(s.mic_id, s.time) for s in new_plan}
        # 到本次重排前已真实执行/尝试过的 (麦,分钟)，不再算"取消"
        happened = {(e.mic_id, e.minute) for e in self._swap_events}
        ad_hoc_mics = {e.mic_id for e in self._swap_events
                       if e.kind == KIND_AD_HOC}

        for e in old_snap.plan:
            key = (e.mic_id, e.time.absolute)
            if key in new_keys or key in happened:
                continue
            cause_code, cause_txt = self._cancel_cause(
                trigger, e.mic_id, e.mic_id in ad_hoc_mics)
            self._cancelled.append(CancelledPlan(
                mic_id=e.mic_id,
                planned_minute=e.time.absolute,
                clock=e.time.clock,
                reason=e.reason,
                removed_in_snapshot=new_seq,
                cause_code=cause_code,
                cause=cause_txt,
            ))

    @staticmethod
    def _cancel_cause(trigger: str, mic_id: str,
                      had_ad_hoc: bool) -> tuple[str, str]:
        if trigger == TRIGGER_DEVICE_REPLACE:
            return CAUSE_DEVICE_REPLACE, "更换机身，计划迁移重排"
        if trigger == TRIGGER_DRAIN_SPIKE:
            return CAUSE_DRAIN_SPIKE, "耗电突增，换电时刻重排"
        if trigger == TRIGGER_AD_HOC_SWAP:
            # 只有发生临时换机的那支麦算"临时换机后旧计划失效"，
            # 其余被连带调整的条目归为临时事件引发的整体重排
            if had_ad_hoc:
                return CAUSE_AD_HOC, f"{mic_id} 临时换机后旧计划失效"
            return CAUSE_REPLAN, "临时换机引发整体重排"
        return CAUSE_REPLAN, "重排后旧计划失效"

    # ---- 终稿 ----
    def _reconcile(self, scene: Scene):
        """按 (绝对分钟, 麦) 把计划与真实执行对账，返回带标注的快照副本。"""
        import copy
        now = scene.now
        # 真实执行事件索引：(分钟, 麦) -> 计划内执行事件
        planned_events: dict[tuple[int, str], SwapEvent] = {}
        for ev in self._swap_events:
            if ev.kind == KIND_PLANNED:
                planned_events.setdefault((ev.minute, ev.mic_id), ev)
        # 该 (麦,分钟) 的计划是否已被取消
        cancel_index = {(c.mic_id, c.planned_minute): c for c in self._cancelled}

        reconciled: list[ReplanSnapshot] = []
        for snap in self._snapshots:
            new_snap = copy.deepcopy(snap)
            for e in new_snap.plan:
                ev = planned_events.get((e.time.absolute, e.mic_id))
                cnl = cancel_index.get((e.mic_id, e.time.absolute))
                if ev is not None:
                    if ev.result == RESULT_SUCCESS:
                        e.execution = ST_EXECUTED
                        e.execution_label = f"已执行（换上 {ev.battery_uid}）"
                        e.battery_uid = ev.battery_uid
                    else:
                        e.execution = ST_FAILED
                        e.execution_label = f"落空（{ev.detail or '无达标备电'}）"
                        e.execution_detail = ev.detail
                elif cnl is not None:
                    e.execution = ST_CANCELLED
                    e.cancel_cause = cnl.cause_code
                    e.execution_label = f"已取消（{cnl.cause}）"
                    e.execution_detail = cnl.cause
                elif e.time.absolute < now:
                    e.execution = ST_NOT_EXECUTED
                    e.execution_label = "未执行（计划时刻已过，无执行记录）"
                else:
                    e.execution = ST_PENDING
                    e.execution_label = "待执行"
            reconciled.append(new_snap)
        return reconciled

    def finalize(self, scene: Scene) -> DispatchReport:
        now = scene.now
        start = (self._start_minute
                 if self._start_minute is not None else now)
        return DispatchReport(
            scene_start=self._tref(start, now),
            horizon=self._tref(scene.horizon, now),
            peak_bands=self._band_clocks(scene),
            finalized_at=self._tref(now, now),
            snapshots=self._reconcile(scene),
            outages=list(self._outages),
            alerts=[{
                "minute": a.minute,
                "clock": clock_of(a.minute, self.day_start_hour),
                "level": a.level,
                "mic_id": a.mic_id,
                "source": a.source,
                "text": a.text,
            } for a in self._alerts],
            swap_events=list(self._swap_events),
            cancelled_plans=list(self._cancelled),
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
            if e.tte_at_swap is None:
                margin = "打烊前不断电"
            elif e.tte_at_swap == 0:
                margin = "换电时已在截止电压"
            else:
                margin = f"不换电余 {e.tte_at_swap:.0f} 分钟"
            lines.append(
                f"     {e.time.clock}（绝对 {e.time.absolute}，{rel}）"
                f" {e.mic_id}  {e.reason}  备电≥{e.min_spare_soc:.0%}{force}"
                f"  [{margin}]")
            # 终稿对账后的真实执行状态（即时渲染时尚未对账则不显示）
            if e.execution and e.execution != ST_PENDING:
                lines.append(f"       └ 执行: {e.execution_label}")
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
        # 该快照计划的终稿对账结果汇总
        n_exec = sum(1 for e in snap.plan if e.execution == ST_EXECUTED)
        n_fail = sum(1 for e in snap.plan if e.execution == ST_FAILED)
        n_cancel = sum(1 for e in snap.plan if e.execution == ST_CANCELLED)
        tail = []
        if n_exec:
            tail.append(f"已执行 {n_exec}")
        if n_fail:
            tail.append(f"落空 {n_fail}")
        if n_cancel:
            tail.append(f"取消 {n_cancel}")
        tail_txt = ("，执行: " + " / ".join(tail)) if tail else ""
        lines.append(
            f"  #{snap.seq} {snap.generated_at.clock} [{snap.trigger}]"
            f" 计划 {len(snap.plan)} 次 / 未解决 {len(snap.unresolved)} 起"
            f" / 校验断电 {snap.validated_outages} 起{tail_txt}"
            + (f" — {snap.note}" if snap.note else ""))

    # 实际换电台账（计划内/临时 × 成功/落空 + 实际电池 UID）
    lines.append(f"\n实际换电台账（共 {len(report.swap_events)} 次）:")
    if report.swap_events:
        for ev in report.swap_events:
            if ev.kind == KIND_AD_HOC:
                kind_txt = "临时换机"
                seq_txt = ""
            else:
                kind_txt = "计划内"
                seq_txt = f" 快照#{ev.plan_seq}" if ev.plan_seq else ""
            if ev.result == RESULT_SUCCESS:
                res_txt = f"✓ 成功换上 {ev.battery_uid}（要求≥{ev.threshold:.0%}）"
            else:
                res_txt = f"✗ 落空：{ev.detail or '无达标备电'}"
            lines.append(
                f"  {ev.clock}（绝对 {ev.minute}，第 {ev.offset_from_start} 分钟）"
                f" {ev.mic_id}  [{kind_txt}{seq_txt}] {res_txt}  ({ev.reason})")
    else:
        lines.append("  （无实际换电动作）")

    # 计划取消台账（旧计划为何失效）
    lines.append(f"\n计划取消台账（共 {len(report.cancelled_plans)} 条）:")
    if report.cancelled_plans:
        for c in report.cancelled_plans:
            lines.append(
                f"  {c.clock}（绝对 {c.planned_minute}） {c.mic_id} "
                f"{c.reason} -> 于快照#{c.removed_in_snapshot} 取消"
                f" [{c.cause}]")
    else:
        lines.append("  ✓ 无旧计划被取消")

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
