"""有限充电位分配（难点二）。

策略
----
READINESS（营业/高峰）: 以"最快产出下一块能上机的电池"为目标。
    候选池里 soc 最高的电池距离可用阈值(80%)最近，优先占用充电位，
    保证任何时刻都有 1~2 块"即取即用"的备电；低电电池让位排队。
BALANCED（歇业后）: 最低电量优先 + 先到先充，轮流深充，利于电池健康。
"""
from __future__ import annotations

from .battery import charge_one_minute as _batt_charge, FULL_SOC
from .models import ChargingPool, SLOT_READINESS

USABLE_SOC = 0.8
EMERGENCY_SOC = 0.5


def assign_slots(pool: ChargingPool) -> None:
    """按当前策略决定哪几块电池占用充电位（每充电分钟前调用）。"""
    charging = [b for b in pool.batteries if b.charging]
    waiting = [b for b in pool.batteries if not b.charging]
    full = [b for b in pool.batteries if b.soc >= FULL_SOC]
    free = pool.slots - len(charging) + len(full)
    # 已充满的电池腾出充电位给排队电池
    for b in full:
        b.charging = False

    if free <= 0:
        return

    if pool.policy == SLOT_READINESS:
        # 最接近可用阈值的先充；接近满电的（马上能取）优先级最高
        waiting.sort(key=lambda b: (-b.soc, b.enqueued_at or 0))
    else:
        waiting.sort(key=lambda b: (b.soc, b.enqueued_at or 0))

    for b in waiting[:free]:
        b.charging = True


def charge_one_minute(pool: ChargingPool) -> None:
    """充电池推进一分钟：先分配充电位，再给在位电池充电。"""
    assign_slots(pool)
    for b in pool.batteries:
        if b.charging:
            _batt_charge(b)


def time_until_usable(pool: ChargingPool, count: int = 1,
                      min_soc: float = USABLE_SOC) -> int | None:
    """影子推演：第几分钟能有 count 块电量 >= min_soc 的备用电池。

    不改动真实充电池；用于 planner 判断"某次换机时有没有备电可用"。
    """
    sim_batts = {id(b): [b.soc, b.charging, b.enqueued_at] for b in pool.batteries}
    # 简化影子对象：用鸭子类型
    class _B:
        __slots__ = ("soc", "charging", "enqueued_at", "uid")

        def __init__(self, src):
            self.soc, self.charging, self.enqueued_at = sim_batts[id(src)]
            self.uid = src.uid

    class _Pool:
        pass

    sim = _Pool()
    sim.slots = pool.slots
    sim.policy = pool.policy
    sim.batteries = [_B(b) for b in pool.batteries]

    minute = 0
    if sum(1 for b in sim.batteries if b.soc >= min_soc) >= count:
        return 0
    while minute < 600:
        charge_one_minute(sim)
        minute += 1
        if sum(1 for b in sim.batteries if b.soc >= min_soc) >= count:
            return minute
    return None
