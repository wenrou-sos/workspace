"""换电计划生成与重排（难点二、三的核心）。

算法（每次重排都从当前现场从零生成，天然支持临时换机后的计划调整）
----------------------------------------------------------------
1. 影子快放：深拷贝现场，按"已确定的换电"从当前时刻快放到打烊，
   逐分钟记录每个麦克风的电量快照与断电时刻。
2. 找最早断电：谁先掉到 CUTOFF_SOC，就先为谁安排换电（截止时间贪心）。
3. 给该麦在断电前的时间窗内逐分钟打分选换机时刻：
     - 落在营业高峰         重罚（避免高峰操作/集中断电）
     - 高峰前 30 分钟       奖励（趁低谷提前换好）
     - 与其他换机间隔太近   惩罚（避免服务员扎堆、备电被瞬时掏空）
4. 关键：候选时刻的可行性不靠"静态快照里有没有电池"，而是把该候选
   真正加进计划后做一次完整重放——只有按顺序执行时确实能取到达标
   备电、且不挤掉其他已排换机，才算可行。这样可以消除"计划各自可行、
   合在一起抢电池"导致的连锁降级。
5. 备电分三档：满电(80%) / 应急半电(50%) / 强制换机(取池中最高电量)，
   模拟"备电紧张也不能让包厢断电"。
6. 重复 1~5 直到无人会断电。
"""
from __future__ import annotations

from .battery import CUTOFF_SOC
from .engine import clone_scene, do_swap, step
from .models import Scene, PlannedSwap

PRE_PEAK_LEAD = 30        # 高峰前多少分钟算"避峰窗口"
PEAK_PENALTY = 60.0       # 落在高峰的代价
PRE_PEAK_BONUS = 18.0     # 避峰窗口奖励
SPREAD = 1.5              # 换机间隔惩罚系数
EARLY_BIAS = 0.02         # 平局打破：在非高峰时段略微倾向早换（让旧电早回池）
USABLE_SOC = 0.8
EMERGENCY_SOC = 0.5
MIN_GAP_SAME_MIC = 90     # 同一支麦两次换机的最小间隔（避免把刚换上的电又换下）


def _in_band(t: int, bands: list[tuple[int, int]]) -> bool:
    return any(a <= t < b for a, b in bands)


def _pre_peak(t: int, bands: list[tuple[int, int]]) -> bool:
    return any(a - PRE_PEAK_LEAD <= t < a for a, _ in bands)


def _simulate(sim: Scene, by_time: dict[int, list[PlannedSwap]]):
    """在影子现场按计划快放，返回 (每分钟换电前麦克风电量, 断电, 执行失败)。"""
    pre_soc: dict[int, dict[str, float]] = {}
    deaths: list[tuple[int, str]] = []
    failed: list[tuple[int, str]] = []

    for t in range(sim.now, sim.horizon):
        pre = {m.mic_id: m.battery.soc for m in sim.mics}
        pre_soc[t] = pre
        for swap in sorted(by_time.get(t, []), key=lambda s: s.mic_id):
            threshold = 0.0 if swap.forced else swap.min_spare_soc
            ok, _ = do_swap(sim, t, swap.mic_id, threshold)
            if not ok:
                failed.append((t, swap.mic_id))
                if sim.pool.batteries:  # 强制兜底：取池中电量最高的电池
                    do_swap(sim, t, swap.mic_id, 0.0)
        step(sim, t)
        for m in sim.mics:
            if m.dead_at == t and pre[m.mic_id] > CUTOFF_SOC:
                deaths.append((t, m.mic_id))
    return pre_soc, deaths, failed


def _replay(scene: Scene, committed: list[PlannedSwap]):
    sim = clone_scene(scene)
    by_time: dict[int, list[PlannedSwap]] = {}
    for s in committed:
        by_time.setdefault(s.time, []).append(s)
    return _simulate(sim, by_time)


def _score(t: int, scene: Scene, committed_times: list[int]) -> float:
    cost = 0.0
    if _in_band(t, scene.peak_bands):
        cost += PEAK_PENALTY
    if _pre_peak(t, scene.peak_bands):
        cost -= PRE_PEAK_BONUS
    # 与既有换机时刻拉开间隔，避免服务员扎堆、备电被瞬时掏空
    nearest = min((abs(t - o) for o in committed_times), default=999)
    if nearest < 30:
        cost += (30 - nearest) * SPREAD
    cost -= (t - scene.now) * EARLY_BIAS  # 同分时倾向早换，旧电池更早回池补能
    return cost


class SwapPlanner:
    def replan(self, scene: Scene) -> tuple[list[PlannedSwap], list[str]]:
        """依据现场当前状态生成完整换电计划，附带调度提示。"""
        committed: list[PlannedSwap] = []
        advisories: list[str] = []

        for _ in range(len(scene.mics) * 3 + 3):
            _, deaths, failed = _replay(scene, committed)
            if failed:
                # 顺序执行时落空的换机（如同一分钟抢同一块备电）：
                # 删掉该条目，按这支麦真正的断电时刻重新安排。
                ft, fmic = failed[0]
                committed = [s for s in committed
                             if not (s.time == ft and s.mic_id == fmic)]
                _, deaths2, _ = _replay(scene, committed)
                death_map = {m: t for t, m in deaths2}
                deadline = death_map.get(fmic)
                if deadline is None:
                    continue  # 该麦其实不需要换机了
                mic_id = fmic
            elif deaths:
                deadline, mic_id = deaths[0]  # 最早断电优先（EDF 贪心）
            else:
                break

            last_swap = max(
                (s.time for s in committed if s.mic_id == mic_id),
                default=scene.now - 1,
            )
            earliest = max(scene.now, last_swap + MIN_GAP_SAME_MIC)
            committed_times = [s.time for s in committed]
            swap = self._pick_swap(
                scene, committed, mic_id, deadline,
                earliest, committed_times, advisories,
            )
            if swap is None:
                advisories.append(
                    f"{mic_id} 将在第 {deadline - scene.now} 分钟后断电，"
                    f"但没有任何可行换机时刻（备电严重不足）")
                break
            committed.append(swap)

        committed.sort(key=lambda s: s.time)
        self._cluster_advice(committed, advisories, scene.now)
        return committed, advisories

    def _pick_swap(self, scene, committed, mic_id, deadline, earliest,
                   committed_times, advisories) -> PlannedSwap | None:
        if deadline <= earliest:
            earliest = scene.now  # 时间窗太紧：放宽同麦间隔限制

        def candidates(threshold):
            # 评分升序，逐个做完整重放验证（真正的全局顺序可行性）：
            # 加入该候选后，整个计划不允许出现任何换机失败，
            # 也不允许制造新的断电（它挤掉别人的电池也算不可行）。
            base_deaths = len(_replay(scene, committed)[1])
            ranked = sorted(
                range(earliest, deadline),
                key=lambda x: _score(x, scene, committed_times),
            )
            for t in ranked:
                trial = PlannedSwap(t, mic_id, "", threshold,
                                    forced=(threshold == 0.0))
                trial_plan = committed + [trial]
                _, new_deaths, failed = _replay(scene, trial_plan)
                if not failed and len(new_deaths) <= base_deaths:
                    yield t

        # 第一档：满电备电
        for t in candidates(USABLE_SOC):
            return PlannedSwap(t, mic_id, "常规", USABLE_SOC)

        # 第二档：应急半电（备电确实紧张时）
        for t in candidates(EMERGENCY_SOC):
            tag = "高峰前半电" if _pre_peak(t, scene.peak_bands) else "半电应急"
            advisories.append(
                f"第 {t - scene.now} 分钟后 {mic_id} 无满电备电可用，"
                f"安排半电(≥{EMERGENCY_SOC:.0%})应急换机")
            return PlannedSwap(t, mic_id, tag, EMERGENCY_SOC)

        # 第三档：断电前强制取池中电量最高的电池
        for t in candidates(0.0):
            advisories.append(
                f"⚠ {mic_id} 预计第 {deadline - scene.now} 分钟后断电，"
                f"届时无达标备电，强制使用池中最高电量电池")
            return PlannedSwap(t, mic_id, "强制(备电不足)", 0.0, forced=True)
        return None

    def _cluster_advice(self, committed, advisories, now: int = 0) -> None:
        times = [s.time for s in committed]
        for s in committed:
            n = sum(1 for o in times if 0 <= o - s.time < 30)
            if n >= 3:
                clock = f"{(s.time // 60) % 24:02d}:{s.time % 60:02d}"
                advisories.append(
                    f"{clock}（第 {s.time - now} 分钟）前后 30 分钟内"
                    f"有 {n} 次换电，注意备电与人力安排")
                break
