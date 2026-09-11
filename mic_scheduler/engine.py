"""离散时间仿真引擎。

时间粒度 = 1 分钟。每分钟顺序：
1) 执行该分钟的换电动作（计划内或临时事件）；
2) 各麦克风按当前使用状态耗电；
3) 记录断电（soc 首次低于 CUTOFF_SOC）；
4) 充电池充电（内部先做充电位再分配）。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .battery import drain_one_minute, CUTOFF_SOC
from .models import Scene, PlannedSwap
from . import pool as pool_mod


@dataclass
class SimResult:
    executed: list[tuple[int, str, str, bool]] = field(default_factory=list)
    # (时间, 麦克风, 备电uid, 是否临时事件)
    failed: list[tuple[int, str]] = field(default_factory=list)
    outages: list[tuple[int, str]] = field(default_factory=list)
    event_log: list[tuple[int, str]] = field(default_factory=list)


def do_swap(scene: Scene, minute: int, mic_id: str,
            min_spare_soc: float) -> tuple[bool, str | None]:
    """执行一次换电：从池中取达标备电，旧电入池。返回 (是否成功, 备电uid)。"""
    mic = scene.mic(mic_id)
    spare = scene.pool.take_best(min_spare_soc)
    if spare is None:
        return False, None
    old = mic.battery
    scene.pool.accept(old, minute)
    mic.battery = spare
    mic.dead_at = None  # 新电池重新计断电时间
    return True, spare.uid


def step(scene: Scene, minute: int) -> list[str]:
    """推进一分钟，返回这一分钟结束时电量已低于截止值的麦克风。

    以电量而非"是否曾经断电"作为判据：换电后电池是新的，同一支麦
    之后仍可能再次断电（一块电池撑不到打烊时本就需要二次换电）。
    """
    low: list[str] = []
    for mic in scene.mics:
        state = mic.state_at(minute)
        drain_one_minute(mic, state)
        if mic.battery.soc <= CUTOFF_SOC:
            if mic.dead_at is None:
                mic.dead_at = minute
            low.append(mic.mic_id)
    pool_mod.charge_one_minute(scene.pool)
    return low


def clone_scene(scene: Scene) -> Scene:
    """深拷贝现场，供 planner 做影子推演（绝不污染真实计划）。"""
    return copy.deepcopy(scene)


def simulate(scene: Scene, plan: list[PlannedSwap],
             events: dict[int, list[tuple[str, float, bool]]] | None = None) -> SimResult:
    """按固定计划（+ 临时换机事件）快进到打烊，用于校验计划质量。

    events: {分钟: [(mic_id, 要求的最低备电电量, 是否必须成功), ...]}
    """
    result = SimResult()
    by_time: dict[int, list[PlannedSwap]] = {}
    for s in plan:
        by_time.setdefault(s.time, []).append(s)
    events = events or {}

    for minute in range(scene.now, scene.horizon):
        # 1) 临时换机事件优先执行（与真实主循环一致：先处理突发，再走计划）
        for mic_id, min_soc, must in events.get(minute, []):
            ok, uid = do_swap(scene, minute, mic_id, min_soc)
            if ok:
                result.executed.append((minute, mic_id, uid, True))
                result.event_log.append((minute, f"临时更换 {mic_id} -> {uid}"))
            elif must:
                result.failed.append((minute, mic_id))
                result.event_log.append((minute, f"⚠ {mic_id} 临时换机失败：无达标备电"))
        # 2) 计划内换电（强制换机兜底取池中能让麦开机的最高电量电池）
        for swap in by_time.get(minute, []):
            threshold = CUTOFF_SOC if swap.forced else swap.min_spare_soc
            ok, uid = do_swap(scene, minute, swap.mic_id, threshold)
            if ok:
                result.executed.append((minute, swap.mic_id, uid, False))
            else:
                result.failed.append((minute, swap.mic_id))
                result.event_log.append(
                    (minute, f"✗ {swap.mic_id} 计划换机无法执行（池中无任何备电）"))
        # 3) 耗电 + 充电；仅登记"当前电池首次断电"
        step(scene, minute)
        for mic in scene.mics:
            if mic.dead_at == minute:
                result.outages.append((minute, mic.mic_id))
    scene.now = scene.horizon
    return result
