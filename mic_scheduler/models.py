"""核心数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 充电位分配策略
SLOT_READINESS = "readiness"   # 高峰/营业中：优先最快产出可用电池
SLOT_BALANCED = "balanced"     # 歇业后：轮流充满，利于电池健康


class MicState(Enum):
    """麦克风工作状态，不同状态耗电不同。"""
    IN_USE = "in_use"   # 正在使用（客人唱歌）
    IDLE = "idle"       # 开机待机
    OFF = "off"         # 关机存放，几乎不耗电


@dataclass
class Battery:
    """一块可更换的充电电池。"""
    uid: str
    soc: float                      # 剩余电量 0~1
    health: float = 1.0             # 容量健康度 0.5~1.0，老化电池掉电更快
    charging: bool = False          # 是否正在充电位上
    enqueued_at: Optional[int] = None  # 进入充电池的时间（用于同优先级先到先充）

    def label(self) -> str:
        return f"{self.uid}({self.soc:.0%})"


@dataclass
class UsageSegment:
    """使用计划时段，时间单位：相对仿真起点的分钟数。"""
    start: int
    end: int
    state: MicState


@dataclass
class Microphone:
    mic_id: str
    battery: Battery
    schedule: list[UsageSegment] = field(default_factory=list)
    drain_scale: float = 1.0        # 设备漏电/电池故障系数，突发故障时调大
    dead_at: Optional[int] = None   # 仿真中实际断电时间（首次）

    def state_at(self, minute: int) -> MicState:
        """查询某分钟的使用状态（时段不重叠；重叠时取后者）。"""
        result = MicState.OFF
        for seg in self.schedule:
            if seg.start <= minute < seg.end:
                result = seg.state
        return result


@dataclass
class ChargingPool:
    """充电柜：slots 个充电位 + 若干块备用电池（含等待/充电中/已充满）。"""
    slots: int
    batteries: list[Battery] = field(default_factory=list)
    policy: str = SLOT_READINESS

    def accept(self, battery: Battery, now: int) -> None:
        """换下来的电池入池排队。"""
        battery.charging = False
        battery.enqueued_at = now
        self.batteries.append(battery)

    def take_best(self, min_soc: float = 0.0) -> Optional[Battery]:
        """取出当前电量最高的、达到 min_soc 的备用电池；没有则 None。"""
        ready = [b for b in self.batteries if b.soc >= min_soc]
        if not ready:
            return None
        ready.sort(key=lambda b: b.soc, reverse=True)
        chosen = ready[0]
        self.batteries.remove(chosen)
        chosen.charging = False
        return chosen

    def ready_count(self, min_soc: float = 0.8) -> int:
        return sum(1 for b in self.batteries if b.soc >= min_soc)


@dataclass
class PlannedSwap:
    """一次计划内换电。"""
    time: int                       # 计划时间（分钟）
    mic_id: str
    reason: str                     # 常规 / 半电 / 强制(预计断电)
    min_spare_soc: float            # 当时备用电池至少要有多少电
    forced: bool = False            # 无法在断电前保证可用电池 -> 硬安排


@dataclass
class Scene:
    """某一时刻的完整现场：麦克风列表 + 充电池 + 营业参数。"""
    mics: list[Microphone]
    pool: ChargingPool
    horizon: int                    # 打烊时间（分钟）
    peak_bands: list[tuple[int, int]]   # 营业高峰时段
    now: int = 0

    def mic(self, mic_id: str) -> Microphone:
        for m in self.mics:
            if m.mic_id == mic_id:
                return m
        raise KeyError(mic_id)
