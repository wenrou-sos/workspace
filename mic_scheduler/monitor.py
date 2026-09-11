"""低电量与断电风险预警（难点四）。

预警不直接改计划，而是把"未来 30~60 分钟会发生什么"告诉调度：
- CRITICAL：某麦预计 30 分钟内断电
- WARNING ：某麦预计 60 分钟内断电 / 备用电池不足 / 高峰前无满电备电
- INFO    ：高峰临近但备电数量充裕等提示（演示中从略）
临时换机、耗电突增后由仿真主循环重新调用本模块，实现"计划调整后预警同步刷新"。
"""
from __future__ import annotations

from dataclasses import dataclass

from .battery import time_to_empty
from .models import Scene
from .pool import USABLE_SOC, time_until_usable

CRITICAL_MIN = 30
WARNING_MIN = 60
SPARES_NEEDED = 2          # 高峰期间希望至少有 2 块满电备电


@dataclass
class Alert:
    minute: int
    level: str               # CRITICAL / WARNING
    text: str
    mic_id: str | None = None    # 设备级预警携带 mic_id，供调度报告关联换电动作；
                                 # 充电池库存类预警为 None
    source: str = "mic"          # mic（单麦断电风险）/ pool（备电库存不足）


class AlertLevel:
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class Monitor:
    def __init__(self):
        self._fired: set[tuple[str, str]] = set()  # (mic, level) 去重

    def reset_dedup(self, mic_id: str | None = None) -> None:
        """临时换机后允许该麦重新触发各级预警。"""
        if mic_id is None:
            self._fired.clear()
        else:
            self._fired = {k for k in self._fired if k[0] != mic_id}

    def check(self, scene: Scene) -> list[Alert]:
        alerts: list[Alert] = []

        # 1) 逐麦预测断电时间（考虑未来使用计划）
        for mic in scene.mics:
            tte = time_to_empty(mic, scene.now, scene.horizon)
            if tte is None:
                continue
            remain = max(0.0, tte - scene.now)
            if remain <= CRITICAL_MIN:
                level = AlertLevel.CRITICAL
            elif remain <= WARNING_MIN:
                level = AlertLevel.WARNING
            else:
                continue
            key = (mic.mic_id, level)
            if key not in self._fired:
                self._fired.add(key)
                alerts.append(Alert(
                    scene.now, level,
                    f"{mic.mic_id} 预计 {remain:.0f} 分钟后断电"
                    f"（当前 {mic.battery.soc:.0%}，"
                    f"健康度 {mic.battery.health:.0%}，"
                    f"耗电系数 {mic.drain_scale:.1f}x）",
                    mic_id=mic.mic_id, source="mic"))

        # 2) 备电库存
        ready = scene.pool.ready_count(USABLE_SOC)
        in_peak = any(a <= scene.now < b for a, b in scene.peak_bands)
        soon_peak = any(0 <= a - scene.now <= 60 for a, _ in scene.peak_bands)
        if (in_peak or soon_peak) and ready < SPARES_NEEDED:
            key = ("POOL", "LOW")
            if key not in self._fired:
                self._fired.add(key)
                wait = time_until_usable(scene.pool, SPARES_NEEDED, USABLE_SOC)
                wait_txt = f"，下一批备电约 {wait} 分钟后可用" if wait else ""
                alerts.append(Alert(
                    scene.now, AlertLevel.WARNING,
                    f"满电备电仅 {ready} 块（高峰要求 {SPARES_NEEDED}）{wait_txt}",
                    source="pool"))
        return alerts
