"""麦克风换电调度系统。

模块划分：
- models:   电池 / 麦克风 / 使用时段 / 充电池 / 场景等数据结构
- battery:  耗电与充电曲线、预计耗尽时间(TTE)
- pool:     有限充电位分配策略
- engine:   离散时间推进（换机事件、耗电、充电）
- planner:  换电计划生成（影子仿真 + 截止时间贪心 + 时段成本评分）
- monitor:  低电量 / 集中断电 / 无可用电池预警
"""

from .models import Battery, Microphone, UsageSegment, MicState, ChargingPool, Scene, PlannedSwap
from .planner import SwapPlanner
from .monitor import Monitor, Alert, AlertLevel

__all__ = [
    "Battery", "Microphone", "UsageSegment", "MicState",
    "ChargingPool", "Scene", "PlannedSwap", "SwapPlanner",
    "Monitor", "Alert", "AlertLevel",
]
