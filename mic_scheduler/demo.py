"""端到端演示：8 支麦克风 + 3 个充电位 + 4 块备电，晚市 18:00 开门。

运行: python -m mic_scheduler.demo
突发事件:
  19:30 M2 客人嫌电量低要求提前换机      -> 临时换机 + 重排
  21:00 M6 设备漏电判定不可用, 换上备用机 -> 换机不换电思路反过来, 计划迁移
  22:30 M5 电池老化突发放电              -> drain_scale 翻倍 + 重排
"""
from __future__ import annotations

import sys

from .battery import CUTOFF_SOC
from .models import (Battery, Microphone, UsageSegment, MicState,
                     ChargingPool, Scene, SLOT_READINESS)
from .engine import do_swap, step
from .planner import SwapPlanner
from .monitor import Monitor
from .report import (DispatchReporter, render_snapshot, render_report,
                     TRIGGER_INITIAL, TRIGGER_AD_HOC_SWAP,
                     TRIGGER_DEVICE_REPLACE, TRIGGER_DRAIN_SPIKE)

OPEN = 18 * 60       # 仿真从 17:00 开始，18:00 开门
CLOSE = 26 * 60      # 凌晨 02:00
START = 17 * 60
PEAK = [(20 * 60, 23 * 60 + 30)]  # 20:00 - 23:30

# (初始电量, 健康度)
INITIAL = {
    "M1": (0.78, 1.00), "M2": (0.62, 1.00), "M3": (0.42, 0.92),
    "M4": (0.36, 0.95), "M5": (0.30, 0.70), "M6": (0.27, 0.88),
    "M7": (0.22, 0.85), "M8": (0.20, 0.80),
}
SPARES = [("S1", 0.95), ("S2", 0.70), ("S3", 0.35), ("S4", 0.10)]
SLOTS = 3

EVENT_SWAP = 19 * 60 + 30
EVENT_REPLACE = 21 * 60
EVENT_SPIKE = 22 * 60 + 30


def hhmm(t: int) -> str:
    t %= 24 * 60
    return f"{t // 60:02d}:{t % 60:02d}"


def build_scene() -> Scene:
    mics: list[Microphone] = []
    for mid, (soc, health) in INITIAL.items():
        segs = [UsageSegment(OPEN, CLOSE, MicState.IN_USE)]
        if mid == "M3":  # 两批客人之间有 1 小时空档，机器待机
            segs = [UsageSegment(OPEN, 21 * 60, MicState.IN_USE),
                    UsageSegment(21 * 60, 22 * 60, MicState.IDLE),
                    UsageSegment(22 * 60, CLOSE, MicState.IN_USE)]
        mics.append(Microphone(mid, Battery(f"B-{mid}", soc, health), segs))

    pool = ChargingPool(
        slots=SLOTS,
        batteries=[Battery(f"B-{uid}", soc) for uid, soc in SPARES],
        policy=SLOT_READINESS,
    )
    for b in pool.batteries:
        b.enqueued_at = START
    return Scene(mics, pool, CLOSE, PEAK, now=START)


def run_demo(export_path: str | None = None) -> None:
    scene = build_scene()
    planner = SwapPlanner()
    monitor = Monitor()
    # 报告器：留存每次重排快照、预警、断电，供调度员事后复核 / 导出
    reporter = DispatchReporter(scene, day_start_hour=0)

    def replan_and_report(trigger: str, note: str = ""):
        """重排 -> 留存结构化快照 -> 打印快照 -> 返回 (计划, 提示)。"""
        plan, advisories = planner.replan(scene)
        snap = reporter.record_replan(scene, plan, advisories, trigger, note)
        print("\n" + render_snapshot(snap))
        return plan, advisories

    plan, advisories = replan_and_report(
        TRIGGER_INITIAL, f"{hhmm(START)} 开班前生成")

    timeline: list[tuple[int, str]] = []
    planned_done: set[tuple[int, str]] = set()

    for minute in range(START, CLOSE):
        scene.now = minute

        # --- 突发事件 1：临时提前换机 ---
        if minute == EVENT_SWAP:
            # 客人坚持要换：优先满电，没有就取池中电量最高的（半电也先满足），
            # 随后立即重排，让系统自己消化这块半电带来的连锁影响。
            threshold = 0.8 if scene.pool.ready_count(0.8) else CUTOFF_SOC
            ok, uid = do_swap(scene, minute, "M2", threshold)
            if ok:
                reporter.mark_rescued("M2")
                kind = "满电" if threshold == 0.8 else "电量最高的备电"
                timeline.append((minute,
                    f"临时换机 M2 -> {uid}（客人提前要求，{kind}）"))
                monitor.reset_dedup("M2")
                plan, advisories = replan_and_report(
                    TRIGGER_AD_HOC_SWAP, f"M2 临时更换为 {uid}（{kind}）")

        # --- 突发事件 2：设备故障，整支换掉（电池不换，转移到备用机身） ---
        if minute == EVENT_REPLACE:
            old = scene.mic("M6")
            spare_body = Microphone("M6'", old.battery,
                                    [seg for seg in old.schedule if seg.end > minute])
            old.battery = Battery(f"B-dead-M6", 0.0, 0.3)
            scene.pool.accept(old.battery, minute)
            scene.mics = [m for m in scene.mics if m.mic_id != "M6"] + [spare_body]
            timeline.append((minute, "M6 机身故障 -> 启用备用机身 M6'（电池转移，计划迁移）"))
            monitor.reset_dedup("M6")
            plan, advisories = replan_and_report(
                TRIGGER_DEVICE_REPLACE, "M6 -> M6'（电池转移，计划迁移）")

        # --- 突发事件 3：M5 电池老化突发放电 ---
        if minute == EVENT_SPIKE:
            m5 = scene.mic("M5")
            m5.drain_scale = 2.2
            timeline.append((minute, "M5 电池异常发热、放电加速（2.2x）"))
            monitor.reset_dedup("M5")
            plan, advisories = replan_and_report(
                TRIGGER_DRAIN_SPIKE, "M5 耗电系数 -> 2.2x")

        # --- 计划内换电 ---
        for s in plan:
            if s.time == minute and (s.time, s.mic_id) not in planned_done:
                planned_done.add((s.time, s.mic_id))
                ok, uid = do_swap(scene, minute, s.mic_id,
                                  CUTOFF_SOC if s.forced else s.min_spare_soc)
                if ok:
                    reporter.mark_rescued(s.mic_id)
                    timeline.append((minute, f"计划换机 {s.mic_id} -> {uid}（{s.reason}）"))
                else:
                    timeline.append((minute, f"✗ {s.mic_id} 换机失败：无任何备电"))

        # --- 耗电 / 充电推进 + 断电登记（仅登记一次） + 预警 ---
        step(scene, minute)
        for mic in scene.mics:
            if mic.dead_at == minute:
                reporter.record_outage(minute, mic.mic_id)
                timeline.append((minute, f"⛡ {mic.mic_id} 已断电！"))
        for alert in monitor.check(scene):
            reporter.record_alert(alert)

    _print_timeline(timeline)
    _print_stats(scene, plan)

    # ---- 终稿调度报告：打印 + 可选导出 JSON ----
    report = reporter.finalize(scene)
    print("\n" + render_report(report))
    if export_path:
        report.to_json(export_path)
        print(f"调度报告已导出: {export_path}")


def _print_timeline(timeline) -> None:
    print("\n=== 关键事件时间线 ===")
    for minute, text in timeline:
        print(f"  {hhmm(minute)}  {text}")


def _print_stats(scene: Scene, plan) -> None:
    print("\n=== 打烊盘点 ===")
    for m in scene.mics:
        print(f"  {m.mic_id:>4} 剩余 {m.battery.soc:4.0%}  电池 {m.battery.uid}")
    print(f"  充电池 {len(scene.pool.batteries)} 块 / {scene.pool.slots} 个充电位:")
    for b in sorted(scene.pool.batteries, key=lambda x: -x.soc):
        print(f"    {b.uid:>8} {b.soc:4.0%}")
    peak_swaps = sum(1 for s in plan
                     if any(a <= s.time < b for a, b in scene.peak_bands))
    print(f"  最终计划 {len(plan)} 次换电，其中高峰时段 {peak_swaps} 次"
          f"（0 次为理想避峰结果）")


if __name__ == "__main__":
    # 用法: python -m mic_scheduler.demo [报告JSON导出路径]
    export = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(export)
