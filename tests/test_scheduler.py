"""调度系统单元测试。运行: python3 -m unittest discover -s tests -v"""
import unittest

from mic_scheduler.models import (Battery, Microphone, UsageSegment, MicState,
                                  ChargingPool, Scene, PlannedSwap,
                                  SLOT_BALANCED, SLOT_READINESS)
from mic_scheduler.battery import (time_to_empty, CUTOFF_SOC,
                                   drain_one_minute, charge_one_minute)
from mic_scheduler.pool import charge_one_minute as pool_tick, USABLE_SOC
from mic_scheduler.engine import simulate, do_swap
from mic_scheduler.planner import SwapPlanner
from mic_scheduler.monitor import Monitor, AlertLevel
from mic_scheduler.report import (DispatchReporter, render_snapshot,
                                  render_report, TRIGGER_INITIAL,
                                  TRIGGER_AD_HOC_SWAP, TRIGGER_DRAIN_SPIKE,
                                  TRIGGER_DEVICE_REPLACE,
                                  KIND_PLANNED, KIND_AD_HOC,
                                  RESULT_SUCCESS, RESULT_FAILED,
                                  ST_EXECUTED, ST_FAILED, ST_CANCELLED,
                                  ST_NOT_EXECUTED, ST_PENDING,
                                  CAUSE_AD_HOC, CAUSE_DEVICE_REPLACE,
                                  CAUSE_DRAIN_SPIKE)
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
        reporter.record_outage(0, "A")
        text = render_report(reporter.finalize(sc))
        self.assertIn("最终断电统计", text)
        self.assertIn("开班已断电 1 起", text)
        self.assertIn("重排记录", text)
        self.assertIn("未解决的备电不足", text)

    def test_rescued_mic_second_outage_is_in_service(self):
        """0% 麦先换入 10% 电池获救，约25分钟后再断电：算营业中断电而非开班断电。

        回归点：初始低电集合只按编号保存、且登记时把事件分钟同时当作
        当前时间（minute <= now 恒真），导致获救后的新断电被多记为
        初始故障、少记营业中断电。
        """
        from mic_scheduler.engine import do_swap, step
        mics = [Microphone("A", Battery("BA", 0.0),
                           [UsageSegment(0, 600, MicState.IN_USE)])]
        sc = scene(mics, spares=(0.10,), slots=1, horizon=600, peak=())
        reporter = DispatchReporter(sc)
        plan, adv = SwapPlanner().replan(sc)
        reporter.record_replan(sc, plan, adv, TRIGGER_INITIAL)

        ok, _ = do_swap(sc, 0, "A", plan[0].min_spare_soc)
        self.assertTrue(ok)
        reporter.mark_rescued("A")  # 成功换入可开机电池

        second = None
        for t in range(0, 600):
            sc.now = t
            step(sc, t)
            if sc.mic("A").dead_at == t:
                second = reporter.record_outage(t, "A")
                break
        self.assertIsNotNone(second)
        self.assertGreater(second.time.absolute, 0)   # 不是开班时刻
        self.assertFalse(second.initial)             # 关键：获救后的新断电

        outages = reporter.finalize(sc).outages
        self.assertEqual(len(outages), 1)
        self.assertFalse(outages[0].initial)
        text = render_report(reporter.finalize(sc))
        self.assertIn("开班已断电 0 起", text)
        self.assertIn("营业中断电 1 起", text)

    def test_initial_outage_without_rescue_stays_initial(self):
        """开班即死且从未获救：无论用哪个 now 登记都保持 initial=True。

        回归点：旧判定 minute <= now 依赖登记时刻；改用显式获救状态后，
        登记时传入的 now 只影响相对时间展示、不影响分类。
        """
        mics = [Microphone("D", Battery("BD", 0.0),
                           [UsageSegment(0, 600, MicState.IN_USE)])]
        sc = scene(mics, spares=(), slots=1, horizon=600, peak=())
        reporter = DispatchReporter(sc)
        # 即便在较晚时刻补登记这条开班断电，分类仍是初始故障
        rec = reporter.record_outage(0, "D", now=120)
        self.assertTrue(rec.initial)
        self.assertEqual(rec.time.relative, -120)  # now 仅用于相对展示

    def test_normal_mic_outage_is_not_initial(self):
        """普通低电麦（开班有电）运营中耗尽：initial=False。"""
        from mic_scheduler.engine import step
        mics = [Microphone("L", Battery("BL", 0.10),
                           [UsageSegment(0, 600, MicState.IN_USE)])]
        sc = scene(mics, spares=(), slots=1, horizon=600, peak=())
        reporter = DispatchReporter(sc)
        for t in range(0, 600):
            sc.now = t
            step(sc, t)
            if sc.mic("L").dead_at == t:
                rec = reporter.record_outage(t, "L")
                self.assertFalse(rec.initial)
                break
        else:
            self.fail("未发生断电")

    def test_tte_at_swap_is_nonnegative_remaining_minutes(self):
        """tte_at_swap 必须是非负剩余分钟（=绝对断电时刻-换电时刻）。

        回归点：① 误用"换电时刻-断电时刻"得到负值（如 -60.2）；
        ② 直接拿当前电量从未来换电时刻积分，漏掉中间耗电而高估续航。
        """
        # 构造"第60分钟换电、不换电约第120分钟断电"：soc0 = .03 + 120/360
        soc0 = CUTOFF_SOC + 120 / 360
        mics = [Microphone("A", Battery("BA", soc0),
                           [UsageSegment(0, 600, MicState.IN_USE)])]
        sc = scene(mics, spares=(1.0, 1.0), slots=2, horizon=600, peak=())
        fake = [PlannedSwap(60, "A", "常规", USABLE_SOC)]
        snap = DispatchReporter(sc).record_replan(
            sc, fake, [], TRIGGER_INITIAL)
        value = snap.plan[0].tte_at_swap
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0)
        self.assertAlmostEqual(value, 60, delta=1)  # 非负、约 60 分钟余量

    def test_tte_at_swap_zero_and_none_edges(self):
        """换电时已在截止电压 -> 0；打烊前不断电 -> None。"""
        # 初始死麦立即换机：剩余 0
        mics = [Microphone("C", Battery("BC", 0.0),
                           [UsageSegment(0, 600, MicState.IN_USE)])]
        sc = scene(mics, spares=(1.0,), slots=1, horizon=600, peak=())
        plan, _ = SwapPlanner().replan(sc)
        snap = DispatchReporter(sc).record_replan(
            sc, plan, [], TRIGGER_INITIAL)
        first = snap.plan[0]
        self.assertEqual(first.time.absolute, 0)
        self.assertEqual(first.tte_at_swap, 0)

        # 短营业 + 满电：打烊前不断电 -> None
        mics2 = [Microphone("B", Battery("BB", 1.0),
                            [UsageSegment(0, 120, MicState.IN_USE)])]
        sc2 = scene(mics2, spares=(1.0,), slots=1, horizon=120, peak=())
        snap2 = DispatchReporter(sc2).record_replan(
            sc2, [PlannedSwap(60, "B", "常规", USABLE_SOC)], [],
            TRIGGER_INITIAL)
        self.assertIsNone(snap2.plan[0].tte_at_swap)

    def test_tte_at_swap_accounts_for_drain_before_swap(self):
        """换电时刻较晚时，剩余余量必须扣减 now->换电时刻之间的耗电。"""
        # 1/3 电量：从现在起约 109 分钟到截止；第 60 分钟换电时只剩约 49
        mics = [Microphone("A", Battery("BA", 1 / 3),
                           [UsageSegment(0, 600, MicState.IN_USE)])]
        sc = scene(mics, spares=(1.0, 1.0), slots=2, horizon=600, peak=())
        snap = DispatchReporter(sc).record_replan(
            sc, [PlannedSwap(60, "A", "常规", USABLE_SOC)], [],
            TRIGGER_INITIAL)
        value = snap.plan[0].tte_at_swap
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0)
        self.assertAlmostEqual(value, 49, delta=1)  # 而非漏掉耗电的 ~109

    # ---- 实际换电台账 / 计划-执行对账 ----
    def test_swap_event_success_records_battery_uid(self):
        """成功的计划内换电在终稿关联到条目，标记 executed 与实际电池 UID。"""
        mics = [mic("A", 0.10, end=600)]
        sc = scene(mics, spares=(1.0,), slots=1)
        rep = DispatchReporter(sc)
        plan = [PlannedSwap(60, "A", "常规", USABLE_SOC)]
        rep.record_replan(sc, plan, [], TRIGGER_INITIAL)
        ok, uid = do_swap(sc, 60, "A", USABLE_SOC)
        self.assertTrue(ok)
        rep.record_swap_execution(60, "A", KIND_PLANNED, RESULT_SUCCESS,
                                  USABLE_SOC, reason="常规", battery_uid=uid)
        # 成功记录内部已自动 mark_rescued
        self.assertIn("A", rep._ever_rescued)
        sc.now = 120
        entry = rep.finalize(sc).snapshots[0].plan[0]
        self.assertEqual(entry.execution, ST_EXECUTED)
        self.assertEqual(entry.battery_uid, uid)
        self.assertIn(uid, entry.execution_label)
        ev = rep.finalize(sc).swap_events[0]
        self.assertEqual(ev.kind, KIND_PLANNED)
        self.assertEqual(ev.result, RESULT_SUCCESS)
        self.assertEqual(ev.minute, 60)

    def test_same_minute_contention_failure_is_distinguished(self):
        """同一分钟两支麦抢一块备电：成功 vs 落空（failed）明确区分，
        落空带失败分钟与原因，且不标记任何电池 UID。"""
        mics = [mic("A", 0.10, end=600), mic("B", 0.11, end=600)]
        sc = scene(mics, spares=(1.0,), slots=1, peak=())
        rep = DispatchReporter(sc)
        plan = [PlannedSwap(60, "A", "常规", USABLE_SOC),
                PlannedSwap(60, "B", "常规", USABLE_SOC)]
        rep.record_replan(sc, plan, [], TRIGGER_INITIAL)
        ok, uid = do_swap(sc, 60, "A", USABLE_SOC)
        rep.record_swap_execution(60, "A", KIND_PLANNED, RESULT_SUCCESS,
                                  USABLE_SOC, "常规", battery_uid=uid)
        ok2, _ = do_swap(sc, 60, "B", USABLE_SOC)
        self.assertFalse(ok2)
        rep.record_swap_execution(60, "B", KIND_PLANNED, RESULT_FAILED,
                                  USABLE_SOC, "常规",
                                  detail="同分钟备电被 A 抢占")
        sc.now = 120
        report = rep.finalize(sc)
        st = {e.mic_id: e for e in report.snapshots[0].plan}
        self.assertEqual(st["A"].execution, ST_EXECUTED)
        self.assertEqual(st["B"].execution, ST_FAILED)
        self.assertIsNone(st["B"].battery_uid)
        self.assertIn("抢占", st["B"].execution_detail)
        failed = [e for e in report.swap_events if e.result == RESULT_FAILED]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].minute, 60)
        self.assertIsNone(failed[0].battery_uid)

    def test_ad_hoc_swap_invalidates_old_plan(self):
        """临时换机记录为 ad_hoc 事件，并把该麦旧计划标为 cancelled(ad_hoc)。"""
        mics = [mic("A", 0.20, end=600), mic("B", 0.40, end=600)]
        sc = scene(mics, spares=(1.0, 0.9), slots=2, peak=())
        rep = DispatchReporter(sc)
        p1, _ = SwapPlanner().replan(sc)
        rep.record_replan(sc, p1, [], TRIGGER_INITIAL)
        a_old = next(s.time for s in p1 if s.mic_id == "A")

        # 第 10 分钟临时给 A 换满电
        ok, uid = do_swap(sc, 10, "A", USABLE_SOC)
        self.assertTrue(ok)
        rep.record_swap_execution(10, "A", KIND_AD_HOC, RESULT_SUCCESS,
                                  USABLE_SOC, "客人提前要求", battery_uid=uid)
        sc.now = 10
        p2, _ = SwapPlanner().replan(sc)
        rep.record_replan(sc, p2, [], TRIGGER_AD_HOC_SWAP, "A 临时换机")

        sc.now = 600
        report = rep.finalize(sc)
        # 临时事件本身在台账中、不绑定快照、带 UID
        ad = [e for e in report.swap_events if e.kind == KIND_AD_HOC]
        self.assertEqual(len(ad), 1)
        self.assertEqual(ad[0].battery_uid, uid)
        self.assertIsNone(ad[0].plan_seq)
        # A 的旧条目被取消且原因码是 ad_hoc
        a_cancel = [c for c in report.cancelled_plans
                    if c.mic_id == "A" and c.planned_minute == a_old]
        self.assertTrue(a_cancel)
        self.assertEqual(a_cancel[0].cause_code, CAUSE_AD_HOC)
        entry = next(e for s in report.snapshots for e in s.plan
                     if s.seq == 1 and e.mic_id == "A"
                     and e.time.absolute == a_old)
        self.assertEqual(entry.execution, ST_CANCELLED)
        self.assertEqual(entry.cancel_cause, CAUSE_AD_HOC)

    def test_cancelled_cause_distinguishes_device_and_spike(self):
        """换机身 -> device_replace；耗电突增 -> drain_spike；普通重排 -> replan。"""
        mics = [mic("A", 0.30, end=600)]
        sc = scene(mics, spares=(1.0, 1.0, 1.0), slots=3, peak=())
        rep = DispatchReporter(sc)
        p1, _ = SwapPlanner().replan(sc)
        rep.record_replan(sc, p1, [], TRIGGER_INITIAL)
        old = next(s.time for s in p1 if s.mic_id == "A")

        # 换机身重排（现场无 ad_hoc 事件）
        sc.now = 100
        rep.record_replan(sc, p1, [], TRIGGER_DEVICE_REPLACE, "换机身")  # 计划未变则无取消
        # 制造一条真实消失的条目：直接给一份新计划（A 换到别的分钟）
        new_plan = [PlannedSwap(old + 50, "A", "常规", USABLE_SOC)]
        rep.record_replan(sc, new_plan, [], TRIGGER_DEVICE_REPLACE, "换机身")
        sc.now = 600
        codes = {c.mic_id: c.cause_code for c in rep.finalize(sc).cancelled_plans}
        self.assertEqual(codes.get("A"), CAUSE_DEVICE_REPLACE)

        # 耗电突增场景单独再来一遍
        sc2 = scene([mic("A", 0.30, end=600)],
                    spares=(1.0, 1.0, 1.0), slots=3, peak=())
        rep2 = DispatchReporter(sc2)
        p21, _ = SwapPlanner().replan(sc2)
        rep2.record_replan(sc2, p21, [], TRIGGER_INITIAL)
        sc2.now = 100
        sc2.mic("A").drain_scale = 3.0
        p22, _ = SwapPlanner().replan(sc2)
        rep2.record_replan(sc2, p22, [], TRIGGER_DRAIN_SPIKE, "2.2x")
        sc2.now = 600
        codes2 = {c.cause_code for c in rep2.finalize(sc2).cancelled_plans}
        self.assertIn(CAUSE_DRAIN_SPIKE, codes2)

    def test_not_executed_and_pending_status(self):
        """计划时刻已过却无执行记录 -> not_executed；时刻未到 -> pending。"""
        mics = [mic("A", 0.30, end=600)]
        sc = scene(mics, spares=(1.0,), slots=1, peak=())
        rep = DispatchReporter(sc)
        rep.record_replan(sc, [PlannedSwap(30, "A", "常规", USABLE_SOC),
                               PlannedSwap(300, "A", "常规", USABLE_SOC)],
                           [], TRIGGER_INITIAL)
        sc.now = 100
        st = {e.time.absolute: e.execution
              for e in rep.finalize(sc).snapshots[0].plan}
        self.assertEqual(st[30], ST_NOT_EXECUTED)
        self.assertEqual(st[300], ST_PENDING)

    def test_swap_ledger_present_in_json(self):
        """JSON 导出含 swap_events / cancelled_plans，字段可读。"""
        mics = [mic("A", 0.10, end=600)]
        sc = scene(mics, spares=(1.0,), slots=1)
        rep = DispatchReporter(sc)
        rep.record_replan(sc, [PlannedSwap(60, "A", "常规", USABLE_SOC)],
                          [], TRIGGER_INITIAL)
        ok, uid = do_swap(sc, 60, "A", USABLE_SOC)
        rep.record_swap_execution(60, "A", KIND_PLANNED, RESULT_SUCCESS,
                                  USABLE_SOC, "常规", battery_uid=uid)
        sc.now = 120
        data = rep.finalize(sc).to_dict()
        self.assertIn("swap_events", data)
        self.assertIn("cancelled_plans", data)
        ev = data["swap_events"][0]
        self.assertEqual(ev["battery_uid"], uid)
        self.assertEqual(ev["kind"], KIND_PLANNED)
        self.assertEqual(ev["result"], RESULT_SUCCESS)
        self.assertEqual(ev["minute"], 60)
        # 既有结构仍在且可读
        self.assertIn("snapshots", data)
        self.assertIn("outages", data)
        json.dumps(data, ensure_ascii=False)  # 可序列化


if __name__ == "__main__":
    unittest.main()
