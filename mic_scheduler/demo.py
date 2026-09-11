"""端到端演示：8 支麦克风 + 3 个充电位 + 4 块备电，晚市 18:00 开门。

运行: python -m mic_scheduler.demo
突发事件:
  19:30 M2 客人嫌电量低要求提前换机      -> 临时换机 + 重排
  21:00 M6 设备漏电判定不可用, 换上备用机 -> 换机不换电思路反过来, 计划迁移
  22:30 M5 电池老化突发放电              -> drain_scale 翻倍 + 重排
"""
from __future__ import annotations

from .battery import time_to_empty
from .models import (Battery, Microphone, UsageSegment, MicState,
                     ChargingPool, Scene, SLOT_READINESS)
from .engine import do_swap, step
from .planner import SwapPlanner
from .monitor import Monitor

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


def print_plan(scene: Scene, plan, advisories, title: str) -> None:
    print(f"\n=== {title} ===")
    if not plan:
        print("  无需换电")
    for s in plan:
        tte = time_to_empty(scene.mic(s.mic_id), s.time, scene.horizon)
        tag = " [高峰!]" if any(a <= s.time < b for a, b in scene.peak_bands) else ""
        forced = " [强制]" if s.forced else ""
        left = f"{tte:.0f} 分钟后将断电" if tte is not None else "打烊前不断电"
        print(f"  {hhmm(s.time)}  {s.mic_id}  ({s.reason}, 备电≥{s.min_spare_soc:.0%})"
              f"  换时{left}{tag}{forced}")
    for a in advisories:
        print(f"  提示: {a}")


def run_demo() -> None:
    scene = build_scene()
    planner = SwapPlanner()
    monitor = Monitor()
    plan, advisories = planner.replan(scene)
    print_plan(scene, plan, advisories,
               f"开班前计划（{hhmm(START)} 生成）：共 {len(plan)} 次换电")

    timeline: list[tuple[int, str]] = []
    planned_done: set[tuple[int, str]] = set()
    alerts_seen: list[tuple[int, str, str]] = []
    outages: list[tuple[int, str]] = []

    for minute in range(START, CLOSE):
        scene.now = minute

        # --- 突发事件 1：临时提前换机 ---
        if minute == EVENT_SWAP:
            # 客人坚持要换：优先满电，没有就取池中电量最高的（半电也先满足），
            # 随后立即重排，让系统自己消化这块半电带来的连锁影响。
            threshold = 0.8 if scene.pool.ready_count(0.8) else 0.0
            ok, uid = do_swap(scene, minute, "M2", threshold)
            if ok:
                kind = "满电" if threshold == 0.8 else "电量最高的备电"
                timeline.append((minute,
                    f"临时换机 M2 -> {uid}（客人提前要求，{kind}）"))
                plan, advisories = planner.replan(scene)
                monitor.reset_dedup("M2")
                print_plan(scene, plan, advisories,
                           f"{hhmm(minute)} 临时换机后重排")

        # --- 突发事件 2：设备故障，整支换掉（电池不换，转移到备用机身） ---
        if minute == EVENT_REPLACE:
            old = scene.mic("M6")
            spare_body = Microphone("M6'", old.battery,
                                    [seg for seg in old.schedule if seg.end > minute])
            old.battery = Battery(f"B-dead-M6", 0.0, 0.3)
            scene.pool.accept(old.battery, minute)
            scene.mics = [m for m in scene.mics if m.mic_id != "M6"] + [spare_body]
            timeline.append((minute, "M6 机身故障 -> 启用备用机身 M6'（电池转移，计划迁移）"))
            plan, advisories = planner.replan(scene)
            monitor.reset_dedup("M6")
            print_plan(scene, plan, advisories,
                       f"{hhmm(minute)} 更换机身后重排")

        # --- 突发事件 3：M5 电池老化突发放电 ---
        if minute == EVENT_SPIKE:
            m5 = scene.mic("M5")
            m5.drain_scale = 2.2
            timeline.append((minute, "M5 电池异常发热、放电加速（2.2x）"))
            plan, advisories = planner.replan(scene)
            monitor.reset_dedup("M5")
            print_plan(scene, plan, advisories,
                       f"{hhmm(minute)} 耗电突增后重排")

        # --- 计划内换电 ---
        for s in plan:
            if s.time == minute and (s.time, s.mic_id) not in planned_done:
                planned_done.add((s.time, s.mic_id))
                ok, uid = do_swap(scene, minute, s.mic_id,
                                  0.0 if s.forced else s.min_spare_soc)
                if ok:
                    timeline.append((minute, f"计划换机 {s.mic_id} -> {uid}（{s.reason}）"))
                else:
                    timeline.append((minute, f"✗ {s.mic_id} 换机失败：无任何备电"))

        # --- 耗电 / 充电推进 + 断电登记（仅登记一次） + 预警 ---
        step(scene, minute)
        for mic in scene.mics:
            if mic.dead_at == minute:
                outages.append((minute, mic.mic_id))
                timeline.append((minute, f"⛡ {mic.mic_id} 已断电！"))
        for alert in monitor.check(scene):
            alerts_seen.append((minute, alert.level, alert.text))

    _print_timeline(timeline, alerts_seen, outages)
    _print_stats(scene, plan)


def _print_timeline(timeline, alerts, outages) -> None:
    print("\n=== 关键事件时间线 ===")
    for minute, text in timeline:
        print(f"  {hhmm(minute)}  {text}")

    print("\n=== 低电量预警（去重后）===")
    for minute, level, text in alerts:
        print(f"  {hhmm(minute)}  [{level}] {text}")

    print("\n=== 断电统计 ===")
    if outages:
        for minute, mic_id in outages:
            print(f"  ✗ {hhmm(minute)}  {mic_id} 营业中断电")
    else:
        print("  ✓ 全场无营业中断电")


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
    run_demo()
