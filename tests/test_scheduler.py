"""调度系统单元测试。运行: python3 -m unittest discover -s tests -v"""
import unittest

from mic_scheduler.models import (Battery, Microphone, UsageSegment, MicState,
                                  ChargingPool, Scene, SLOT_BALANCED, SLOT_READINESS)
from mic_scheduler.battery import (time_to_empty, CUTOFF_SOC,
                                   drain_one_minute, charge_one_minute)
from mic_scheduler.pool import charge_one_minute as pool_tick, USABLE_SOC
from mic_scheduler.engine import simulate, do_swap
from mic_scheduler.planner import SwapPlanner
from mic_scheduler.monitor import Monitor, AlertLevel


def mic(mid, soc, start=0, end=600, health=1.0, state=MicState.IN_USE):
    return Microphone(mid, Battery(f"B-{mid}", soc, health),
                      [UsageSegment(start, end, state)])


def scene(mics, spares=(), slots=3, horizon=600, peak=((300, 420),), now=0):
    pool = ChargingPool(slots,
                        [Battery(f"S{i}", s) for i, s in enumerate(spares)])
    for b in pool.batteries:
        b.enqueued_at = now
    return Scene(mics, pool, horizon, list(peak), now=now)


class BatteryModelTest(unittest.TestCase):
    def test_time_to_empty_full_battery_in_use(self):
        m = mic("M", 1.0)
        # 满电连续使用约 360 分钟到 0，349 分钟左右到 3% 截止
        tte = time_to_empty(m, 0, 600)
        self.assertAlmostEqual(tte, (1 - CUTOFF_SOC) * 360, delta=1.0)

    def test_predict_soc_and_health(self):
        m1 = mic("A", 1.0)
        m2 = mic("B", 1.0, health=0.5)
        for _ in range(60):
            drain_one_minute(m1, MicState.IN_USE)
            drain_one_minute(m2, MicState.IN_USE)
        self.assertGreater(m1.battery.soc, m2.battery.soc)
        self.assertAlmostEqual(m1.battery.soc, 1 - 60 / 360, delta=0.01)

    def test_charging_cv_slowdown(self):
        b = Battery("X", 0.0)
        for _ in range(80):
            charge_one_minute(b)
        self.assertGreaterEqual(b.soc, 0.8)
        soc_at_80 = b.soc
        for _ in range(10):
            charge_one_minute(b)
        gained = b.soc - soc_at_80
        self.assertLess(gained, 0.2)  # CV 阶段明显变慢

    def test_off_state_barely_drains(self):
        m = Microphone("O", Battery("B", 1.0),
                       [UsageSegment(0, 600, MicState.OFF)])
        self.assertIsNone(time_to_empty(m, 0, 600))


class ChargingPoolTest(unittest.TestCase):
    def test_readiness_prefers_near_full_battery(self):
        pool = ChargingPool(1, [Battery("lo", 0.1), Battery("hi", 0.7)],
                            SLOT_READINESS)
        for b in pool.batteries:
            b.enqueued_at = 0
        pool_tick(pool)
        self.assertTrue(next(b for b in pool.batteries if b.uid == "hi").charging)
        self.assertFalse(next(b for b in pool.batteries if b.uid == "lo").charging)

    def test_balanced_prefers_lowest(self):
        pool = ChargingPool(1, [Battery("lo", 0.1), Battery("hi", 0.7)],
                            SLOT_BALANCED)
        for b in pool.batteries:
            b.enqueued_at = 0
        pool_tick(pool)
        self.assertTrue(next(b for b in pool.batteries if b.uid == "lo").charging)

    def test_full_battery_frees_slot(self):
        pool = ChargingPool(1, [Battery("f", 0.999), Battery("w", 0.2)],
                            SLOT_READINESS)
        for b in pool.batteries:
            b.enqueued_at = 0
        pool_tick(pool)
        self.assertTrue(next(b for b in pool.batteries if b.uid == "w").charging)


class PlannerTest(unittest.TestCase):
    def test_plan_avoids_outages(self):
        sc = scene([mic(f"M{i}", s, end=480) for i, s in
                    enumerate([0.2, 0.25, 0.3, 0.35])],
                   spares=(1.0, 1.0, 0.9), slots=3, horizon=480)
        plan, adv = SwapPlanner().replan(sc)
        sc2 = scene([mic(f"M{i}", s, end=480) for i, s in
                     enumerate([0.2, 0.25, 0.3, 0.35])],
                    spares=(1.0, 1.0, 0.9), slots=3, horizon=480)
        result = simulate(sc2, plan)
        self.assertEqual(result.outages, [], f"发生断电: {result.outages}")
        self.assertEqual(result.failed, [], f"换机失败: {result.failed}")

    def test_no_swaps_when_batteries_last(self):
        sc = scene([mic("M1", 1.0, end=120)], slots=1, horizon=120)
        plan, _ = SwapPlanner().replan(sc)
        self.assertEqual(plan, [])

    def test_same_minute_contention_resolved(self):
        # 两支几乎同电的麦 + 1 块满电备电：计划不得把两支排在同一分钟
        sc = scene([mic("A", 0.10, end=400), mic("B", 0.11, end=400)],
                   spares=(1.0, 0.4), slots=1, horizon=400, peak=())
        plan, _ = SwapPlanner().replan(sc)
        times = {}
        for s in plan:
            times.setdefault(s.time, []).append(s.mic_id)
        self.assertTrue(all(len(v) == 1 for v in times.values()))
        # 用全新等价场景验证计划可执行、不断电
        sc2 = scene([mic("A", 0.10, end=400), mic("B", 0.11, end=400)],
                    spares=(1.0, 0.4), slots=1, horizon=400, peak=())
        r = simulate(sc2, plan)
        self.assertEqual(r.failed, [])
        self.assertEqual(r.outages, [])

    def test_forced_swap_when_spares_too_tight(self):
        # 3 支低电麦、2 块备电、1 个充电位、营业极长 -> 必然紧张，
        # 应出现强制换机而不是抛异常，且最终断电数应最少
        sc = scene([mic("A", 0.08), mic("B", 0.09), mic("C", 0.10)],
                   spares=(0.6, 0.2), slots=1, horizon=700, peak=((300, 420),))
        plan, adv = SwapPlanner().replan(sc)
        self.assertTrue(any(s.forced for s in plan) or adv)

    def test_replan_after_ad_hoc_swap(self):
        """临时换机后重排：旧计划作废，基于新现场仍应无断电。"""
        sc = scene([mic(f"M{i}", s) for i, s in
                    enumerate([0.15, 0.3, 0.45])],
                   spares=(1.0, 0.9), slots=2)
        _plan1, _ = SwapPlanner().replan(sc)
        # 第 10 分钟服务员临时给 M0 换一块满电（真实事件），随后重排
        ok, uid = do_swap(sc, 10, "M0", USABLE_SOC)
        self.assertTrue(ok)
        sc.now = 10
        plan2, _ = SwapPlanner().replan(sc)
        self.assertTrue(all(s.time >= 10 for s in plan2))
        # 临时换机已经真实发生过：在这个现场上直接执行新计划，应无断电、无落空
        r = simulate(sc, plan2)
        self.assertEqual(r.failed, [])
        self.assertEqual(r.outages, [])

    def test_initial_dead_mic_is_registered_at_creation(self):
        """场景创建时电量已低于保护截止电压：立即登记为断电，而不是 None。"""
        m = mic("DEAD", 0.0)
        sc = scene([m], spares=(1.0,))
        self.assertEqual(m.dead_at, 0)
        m2 = mic("DEAD2", CUTOFF_SOC)
        sc2 = scene([m2], now=50, spares=(1.0,))
        self.assertEqual(m2.dead_at, 50)

    def test_initial_dead_mic_gets_immediate_swap(self):
        """开班即低电的麦必须在第 0 分钟安排换机，且计划执行后零断电。"""
        build = lambda: scene(
            [mic("D0", 0.0, end=600), mic("OK", 0.50, end=600)],
            spares=(1.0,), slots=1, horizon=600, peak=())
        plan, advisories = SwapPlanner().replan(build())
        first = [s for s in plan if s.mic_id == "D0"]
        self.assertTrue(first, f"开班即断电的麦未被排程: {advisories}")
        self.assertEqual(first[0].time, 0)
        r = simulate(build(), plan)
        self.assertEqual(r.outages, [])
        self.assertEqual(r.failed, [])

    def test_initial_dead_two_mics_one_spare(self):
        """两支开班即断电、只有一块满电：一支立即得救，另一支如实告警登记断电。"""
        build = lambda: scene(
            [mic("D0", 0.0, end=300), mic("D1", 0.0, end=300)],
            spares=(1.0,), slots=1, horizon=300, peak=())
        plan, advisories = SwapPlanner().replan(build())
        self.assertEqual([(s.time, s.mic_id) for s in plan], [(0, "D0")])
        self.assertTrue(any("截止电压" in a and "没有任何可换电池" in a
                            for a in advisories))
        r = simulate(build(), plan)
        self.assertEqual(r.outages, [(0, "D1")])

    def test_junk_spare_is_not_fake_rescue(self):
        """池中唯一备电同样低于截止电压：不允许死电换死电的无效/重复换机。"""
        build = lambda: scene([mic("D0", 0.0, end=600)],
                              spares=(CUTOFF_SOC * 0.5,), slots=1,
                              horizon=600, peak=())
        plan, advisories = SwapPlanner().replan(build())
        self.assertEqual(plan, [])
        self.assertTrue(any("截止电压" in a for a in advisories))
        r = simulate(build(), plan)
        self.assertEqual(r.outages, [(0, "D0")])

    def test_weak_rescue_then_second_swap(self):
        """立即换入的电池电量不高（10%）：先救急，之后再次断电时继续安排换机。"""
        build = lambda: scene([mic("D0", 0.0, end=600)],
                              spares=(0.10,), slots=1, horizon=600, peak=())
        plan, _ = SwapPlanner().replan(build())
        self.assertTrue(plan)
        self.assertEqual(plan[0].time, 0)
        self.assertTrue(plan[0].forced)
        self.assertGreaterEqual(len(plan), 2)  # 弱电池撑不到打烊，需二次换机
        r = simulate(build(), plan)
        self.assertEqual(r.outages, [])
        self.assertEqual(r.failed, [])

    def test_replan_when_mic_already_dead_mid_run(self):
        """重排发生在某麦实际断电之后：也必须在当前时刻立即安排救援。"""
        sc = scene([mic("D0", 0.10, end=600), mic("OK", 0.90, end=600)],
                   spares=(1.0,), slots=1, horizon=600, peak=())
        from mic_scheduler.engine import step
        for t in range(0, 60):
            step(sc, t)
        sc.now = 60
        self.assertIsNotNone(sc.mic("D0").dead_at)
        self.assertLess(sc.mic("D0").dead_at, 60)
        plan, _ = SwapPlanner().replan(sc)
        d0 = [s for s in plan if s.mic_id == "D0"]
        self.assertTrue(d0)
        self.assertEqual(d0[0].time, 60)


class MonitorTest(unittest.TestCase):
    def test_critical_and_warning_levels(self):
        sc = scene([mic("DANGER", 0.10)], slots=0, horizon=600)
        mon = Monitor()
        # 0.10 电量约 36 分钟到 0，25 分钟到 3% -> 进入 30 分钟临界窗
        sc.now = 0
        alerts = mon.check(sc)
        self.assertTrue(any(a.level == AlertLevel.CRITICAL for a in alerts))

    def test_dedup_and_reset(self):
        sc = scene([mic("X", 0.10)], slots=0)
        mon = Monitor()
        first = mon.check(sc)
        second = mon.check(sc)
        self.assertTrue(first)
        self.assertFalse(second)  # 去重
        mon.reset_dedup("X")
        self.assertTrue(mon.check(sc))  # 换机后重新可报

    def test_low_spare_alert_near_peak(self):
        sc = scene([mic("X", 1.0, end=600)], spares=(0.85,), slots=1,
                   horizon=600, peak=((30, 120),))
        sc.now = 0  # 距高峰 30 分钟，只有 1 块满电（要求 2）
        alerts = Monitor().check(sc)
        self.assertTrue(any("备电" in a.text for a in alerts))


if __name__ == "__main__":
    unittest.main()
