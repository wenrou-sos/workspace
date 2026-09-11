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
from mic_scheduler.report import (DispatchReporter, render_snapshot,
                                  render_report, TRIGGER_INITIAL,
                                  TRIGGER_AD_HOC_SWAP, TRIGGER_DRAIN_SPIKE)
import json


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


def _replan_snapshot(mics, spares=(), trigger=TRIGGER_INITIAL,
                     now=0, horizon=600, peak=(), slots=2, note="",
                     reporter=None, alerts=()):
    """构造场景 -> 重排 -> 留存快照，返回 (scene, plan, advisories, reporter, snapshot)。"""
    sc = scene(mics, spares=spares, slots=slots, horizon=horizon,
               peak=peak, now=now)
    if reporter is None:
        reporter = DispatchReporter(sc)
    for a in alerts:
        reporter.record_alert(a)
    plan, adv = SwapPlanner().replan(sc)
    snap = reporter.record_replan(sc, plan, adv, trigger, note)
    return sc, plan, adv, reporter, snap


class ReportTest(unittest.TestCase):
    def test_snapshot_contains_all_sections(self):
        """报告快照包含场景时间/TTE/计划(原因+阈值)/校验统计。"""
        sc, plan, _, _, snap = _replan_snapshot(
            [mic("A", 0.20, end=480), mic("B", 0.30, end=480)],
            spares=(1.0, 1.0), slots=2, horizon=480)
        self.assertEqual(snap.seq, 1)
        self.assertEqual(snap.trigger, TRIGGER_INITIAL)
        self.assertEqual(snap.generated_at.absolute, 0)
        self.assertEqual(snap.generated_at.relative, 0)
        self.assertEqual(snap.horizon.absolute, 480)
        self.assertEqual(snap.horizon.relative, 480)
        # 两支麦都有 TTE 预测
        by_mic = {f.mic_id: f for f in snap.forecasts}
        self.assertIn("A", by_mic)
        self.assertIsNotNone(by_mic["A"].tte_relative)
        self.assertFalse(by_mic["A"].initial_dead)
        # 计划条目带原因与备电阈值
        self.assertTrue(snap.plan)
        self.assertTrue(all(e.reason and e.min_spare_soc > 0
                            for e in snap.plan))
        # 计划可执行：零断电、零落空
        self.assertEqual(snap.validated_outages, 0)
        self.assertEqual(snap.validated_failed, 0)
        self.assertEqual(snap.unresolved, [])

    def test_time_dual_track_absolute_and_relative(self):
        """now=100 时生成快照：绝对分钟与相对分钟必须同时给出且相差 100。"""
        m = mic("A", 0.20, start=100, end=400)
        sc = scene([m], spares=(1.0,), slots=1, horizon=400, now=100)
        reporter = DispatchReporter(sc)
        plan, adv = SwapPlanner().replan(sc)
        snap = reporter.record_replan(sc, plan, adv, TRIGGER_AD_HOC_SWAP)
        for e in snap.plan:
            self.assertEqual(e.time.absolute - e.time.relative, 100)
            self.assertGreaterEqual(e.time.absolute, 100)
        f = next(f for f in snap.forecasts if f.mic_id == "A")
        self.assertEqual(f.tte_absolute - f.tte_relative, 100)

    def test_initial_dead_forecast_and_unresolved(self):
        """开班即断电且无可用备电：TTE 相对为 0、initial_dead=True、列入未解决。"""
        _, plan, adv, _, snap = _replan_snapshot(
            [mic("DEAD", 0.0, end=600)], spares=(0.02,), slots=1)
        self.assertEqual(plan, [])
        f = snap.forecasts[0]
        self.assertTrue(f.initial_dead)
        self.assertEqual(f.tte_relative, 0)
        self.assertEqual(f.related_swap, None)
        self.assertEqual(len(snap.unresolved), 1)
        u = snap.unresolved[0]
        self.assertEqual(u.mic_id, "DEAD")
        self.assertTrue(u.initial_dead and u.outage)
        self.assertEqual(u.outage_time.absolute, 0)
        self.assertGreaterEqual(snap.validated_outages, 1)
        # 渲染文本保留"截止电压"与初始断电标记
        text = render_snapshot(snap)
        self.assertIn("截止电压", text)
        self.assertIn("开班已低于截止电压", text)

    def test_failure_snapshot_is_retained_not_dropped(self):
        """无备电失败场景的快照也必须留存，且终稿包含全部快照。"""
        reporter = None
        _, _, _, reporter, s1 = _replan_snapshot(
            [mic("A", 0.0, end=600)], spares=(), slots=1, reporter=reporter)
        _, _, _, reporter, s2 = _replan_snapshot(
            [mic("A", 0.0, end=600)], spares=(1.0,), slots=1,
            trigger=TRIGGER_AD_HOC_SWAP, note="补来一块满电", reporter=reporter)
        report = reporter.finalize(
            scene([mic("A", 0.0, end=600)], spares=(1.0,), slots=1))
        self.assertEqual(len(report.snapshots), 2)
        self.assertEqual(report.snapshots[0].seq, 1)
        self.assertTrue(report.snapshots[0].unresolved)  # 失败快照保留
        self.assertEqual(report.snapshots[1].validated_outages, 0)

    def test_replan_diff_detects_shift_added_removed(self):
        """临时换机后重排：差异要识别时刻位移、新增、取消（按绝对分钟对齐）。"""
        mics = lambda: [mic("A", 0.15, end=600), mic("B", 0.45, end=600)]
        sc = scene(mics(), spares=(1.0, 0.9), slots=2, horizon=600)
        reporter = DispatchReporter(sc)
        p1, _ = SwapPlanner().replan(sc)
        s1 = reporter.record_replan(sc, p1, [], TRIGGER_INITIAL)

        ok, _ = do_swap(sc, 10, "A", USABLE_SOC)
        self.assertTrue(ok)
        sc.now = 10
        p2, _ = SwapPlanner().replan(sc)
        s2 = reporter.record_replan(sc, p2, [], TRIGGER_AD_HOC_SWAP,
                                    "A 客人要求提前换机")

        d = reporter.diff(s1, s2)
        # A 被临时换机救走：它原来的换电条目必然取消或位移
        a_changes = [c for c in d.changed if c["mic_id"] == "A"]
        a_removed = [e for e in d.removed if e.mic_id == "A"]
        self.assertTrue(a_changes or a_removed)
        # 差异里所有时间均为绝对分钟
        for c in d.changed:
            self.assertIn("shift_minutes", c)
            self.assertEqual(c["new"]["absolute"] - c["old"]["absolute"],
                             c["shift_minutes"])

    def test_drain_spike_moves_swap_earlier(self):
        """耗电突增后重排：同一支麦的换电时刻应提前（shift 为负）。"""
        m = mic("A", 0.50, end=600)
        sc = scene([m, mic("B", 0.90, end=600)],
                   spares=(1.0, 1.0, 1.0), slots=3, horizon=600)
        reporter = DispatchReporter(sc)
        p1, _ = SwapPlanner().replan(sc)
        s1 = reporter.record_replan(sc, p1, [], TRIGGER_INITIAL)
        before = next(e.time.absolute for e in s1.plan if e.mic_id == "A")

        sc.now = 100
        sc.mic("A").drain_scale = 3.0
        p2, _ = SwapPlanner().replan(sc)
        s2 = reporter.record_replan(sc, p2, [], TRIGGER_DRAIN_SPIKE)
        after_entries = [e for e in s2.plan if e.mic_id == "A"]
        self.assertTrue(after_entries)
        self.assertLess(after_entries[0].time.absolute, before + 100)
        # 快照历史均保留
        self.assertEqual(len(reporter.finalize(sc).snapshots), 2)

    def test_alert_linked_to_swap_action(self):
        """设备级预警应关联到该麦的首次换电条目（related_alert 非空）。"""
        from mic_scheduler.monitor import Alert, AlertLevel
        sc = scene([mic("A", 0.10, end=600)], spares=(1.0,), slots=1)
        reporter = DispatchReporter(sc)
        reporter.record_alert(Alert(0, AlertLevel.CRITICAL,
                                    "A 预计 25 分钟后断电",
                                    mic_id="A", source="mic"))
        plan, adv = SwapPlanner().replan(sc)
        snap = reporter.record_replan(sc, plan, adv, TRIGGER_INITIAL)
        entry = next(e for e in snap.plan if e.mic_id == "A")
        self.assertIsNotNone(entry.related_alert)
        self.assertIn("A", entry.related_alert)
        forecast = next(f for f in snap.forecasts if f.mic_id == "A")
        self.assertIsNotNone(forecast.related_swap)

    def test_report_json_export_roundtrip(self):
        """终稿报告可导出 JSON，结构含全部必需字段且可反序列化。"""
        import tempfile, os
        # A 开班即低电且池中只有死电：校验确实在第 0 分钟断电
        sc, _, _, reporter, _ = _replan_snapshot(
            [mic("A", 0.0, end=300)], spares=(0.02,), slots=1, horizon=300,
            peak=((200, 300),))
        reporter.record_outage(0, "A", 0)  # 开班已断电
        report = reporter.finalize(sc)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.json")
            report.to_json(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        for key in ("scene_start", "horizon", "peak_bands", "finalized_at",
                    "snapshots", "outages", "alerts"):
            self.assertIn(key, data)
        self.assertEqual(data["outages"][0]["mic_id"], "A")
        self.assertEqual(data["outages"][0]["initial"], True)
        # 无可用备电 -> 无计划但快照仍留存，时间字段双轨一致
        snap0 = data["snapshots"][0]
        self.assertEqual(snap0["plan"], [])
        self.assertEqual(snap0["generated_at"]["absolute"],
                         snap0["generated_at"]["relative"])

    def test_render_report_has_outage_stats(self):
        """终稿文本含断电统计（开班/营业中分类）与全部快照清单。"""
        sc, _, _, reporter, _ = _replan_snapshot(
            [mic("A", 0.0, end=300)], spares=(0.02,), slots=1, horizon=300)
        # 校验中断电发生在第 0 分钟（开班即死），手动登记实际断电
        reporter.record_outage(0, "A", 0)
        text = render_report(reporter.finalize(sc))
        self.assertIn("最终断电统计", text)
        self.assertIn("开班已断电 1 起", text)
        self.assertIn("重排记录", text)
        self.assertIn("未解决的备电不足", text)


if __name__ == "__main__":
    unittest.main()
