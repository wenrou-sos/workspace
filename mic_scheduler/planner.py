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
5. 备电分三档：满电(80%) / 应急半电(50%) / 强制换机(取池中电量最高、
   且高于截止电压、能让麦真正开机的电池)，模拟"备电紧张也不能让包厢断电"；
   低于截止电压的死电不具备救援资格，防止"死电换死电"的假救援被排进计划。
   开班（场景创建）时就已低于截止电压的麦由 Scene 直接登记为即时断电，
   在这里得到 t=now 的立即换电安排。
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

    # 重排可能发生在某支麦实际断电之后（仿真已推进、现场直接重建等情况）：
    # 此刻仍低于截止电压且未被换走的麦，按"现在就要救"处理，
    # 否则它的 dead_at 是过去时刻，永远不会出现在快放的断电事件里。
    for m in sim.mics:
        if (m.dead_at is not None and m.dead_at <= sim.now
                and m.battery.soc <= CUTOFF_SOC):
            m.dead_at = sim.now

    # 每支麦"快放中是否曾在阈值之上活着"：开班即低电且从未被救活的麦，
    # 其初始断电只登记一次——若第 0 分钟换机换进来的同样是死电（假救援），
    # 不能反复当成新断电给同一支麦重复排换机；真正救活后再次跌破的，
    # 属于正常的新断电事件，照常入 EDF 队列。
    ever_alive = {m.mic_id: m.battery.soc > CUTOFF_SOC for m in sim.mics}

    for t in range(sim.now, sim.horizon):
        pre = {m.mic_id: m.battery.soc for m in sim.mics}
        pre_soc[t] = pre
        for swap in sorted(by_time.get(t, []), key=lambda s: s.mic_id):
            # 强制换机取池中能让麦开机的最高电量电池：阈值是截止电压而非 0，
            # 否则换进来一块同样低于截止电压的死电，等于没有救援
            threshold = CUTOFF_SOC if swap.forced else swap.min_spare_soc
            ok, _ = do_swap(sim, t, swap.mic_id, threshold)
            if not ok:
                failed.append((t, swap.mic_id))
                if sim.pool.batteries:  # 强制兜底：取池中电量最高的电池
                    do_swap(sim, t, swap.mic_id, CUTOFF_SOC)
        step(sim, t)
        for m in sim.mics:
            if m.battery.soc > CUTOFF_SOC:
                ever_alive[m.mic_id] = True
                continue  # 这一分钟的换机把它救活了（或本就活着）
            if m.dead_at != t:
                continue
            if ever_alive[m.mic_id]:
                # 曾经活着 -> 本分钟首次跌破截止电压
                deaths.append((t, m.mic_id))
            elif t == sim.now:
                # 快放开始时就已断电、且立即换机也没救活：初始登记一次
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
                if deadline <= scene.now:
                    advisories.append(
                        f"{mic_id} 电量已低于保护截止电压，"
                        f"但没有任何可换电池（备电严重不足）")
                else:
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
        already_dead = deadline <= scene.now
        if already_dead:
            # 现场创建时（或重排时刻）就已低于截止电压：只能立即换机，
            # 不再受同麦换机间隔限制
            earliest = scene.now
        elif deadline <= earliest:
            earliest = scene.now  # 时间窗太紧：放宽同麦间隔限制

        window = ([scene.now] if already_dead
                  else list(range(earliest, deadline)))
        if not window:
            return None

        def candidates(threshold):
            # 评分升序，逐个做完整重放验证（真正的全局顺序可行性）。
            # 可行条件：
            #  1) 整个计划没有任何换机失败；
            #  2) 其他麦不得出现原计划没有的断电时刻（不能靠抢别人的
            #     电池救这支麦）；
            #  3) 目标麦必须真正脱离本次断电——换入低于截止电压的死电
            #     （死电换死电）时它仍在 t 时刻断电，属"假救援"，拒绝，
            #     否则同一支麦会在同一分钟被反复安排无效换机；
            #  4) 目标麦换电后不得比原截止时刻更早断电（防止换进一块
            #     比它自身余电还差的电池反而缩短续航）。
            _, base_deaths, _ = _replay(scene, committed)
            base_by: dict[str, set[int]] = {}
            for dt, dm in base_deaths:
                base_by.setdefault(dm, set()).add(dt)
            ranked = sorted(
                window,
                key=lambda x: _score(x, scene, committed_times),
            )
            for t in ranked:
                trial = PlannedSwap(t, mic_id, "", threshold, forced=False)
                _, new_deaths, failed = _replay(scene, committed + [trial])
                if failed:
                    continue
                new_by: dict[str, set[int]] = {}
                for dt, dm in new_deaths:
                    new_by.setdefault(dm, set()).add(dt)
                target_times = new_by.pop(mic_id, set())
                if t in target_times:
                    continue  # 目标麦在 t 仍断电：假救援
                if any(dt < deadline for dt in target_times):
                    continue  # 换入电池比自身余电还差，死得更早
                if any(not times <= base_by.get(other, set())
                       for other, times in new_by.items()):
                    continue  # 挤掉别的麦的电池，制造了新断电
                yield t

        # 第一档：满电备电
        for t in candidates(USABLE_SOC):
            reason = "立即换电" if already_dead and t == scene.now else "常规"
            return PlannedSwap(t, mic_id, reason, USABLE_SOC)

        # 第二档：应急半电（备电确实紧张时）
        for t in candidates(EMERGENCY_SOC):
            tag = "高峰前半电" if _pre_peak(t, scene.peak_bands) else "半电应急"
            advisories.append(
                f"第 {t - scene.now} 分钟后 {mic_id} 无满电备电可用，"
                f"安排半电(≥{EMERGENCY_SOC:.0%})应急换机")
            return PlannedSwap(t, mic_id, tag, EMERGENCY_SOC)

        # 第三档：断电前强制取池中电量最高、且能让麦开机的电池
        for t in candidates(CUTOFF_SOC):
            if already_dead:
                advisories.append(
                    f"⚠ {mic_id} 电量已低于保护截止电压(≤{CUTOFF_SOC:.0%})，"
                    f"立即强制使用池中最高电量电池")
            else:
                advisories.append(
                    f"⚠ {mic_id} 预计第 {deadline - scene.now} 分钟后断电，"
                    f"届时无达标备电，强制使用池中最高电量电池")
            return PlannedSwap(t, mic_id, "强制(备电不足)", CUTOFF_SOC,
                               forced=True)
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
