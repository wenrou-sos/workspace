"""电池耗电与充电模型（难点一：电量消耗模拟）。

建模说明
--------
1. 耗电按状态分流：使用中 / 待机 / 关机三档，关机几乎不掉电；
   单块电池再乘 health（老化）和设备 drain_scale（漏电、突发故障）。
2. 这里采用分段线性耗电（KTV 麦克风数小时场景足够准确）；
   若需要更真实的锂电曲线，可在 ``drain_rate`` / ``charge_rate``
   中替换为按 soc 查表的非线性曲线，接口不变。
3. 充电采用先恒流后恒压：0~80% 快充，80% 以上逐渐变慢。
"""
from __future__ import annotations

from .models import Battery, Microphone, MicState

# 每分钟的电量消耗（满电=1）
DRAIN_PER_MIN = {
    MicState.IN_USE: 1.0 / 360.0,  # 连续使用约 6 小时
    MicState.IDLE: 1.0 / 1440.0,   # 待机约 24 小时
    MicState.OFF: 1.0 / 2880.0,    # 关机存放
}

# 每分钟可充入的电量（CC 阶段 / CV 阶段）
CHARGE_CC_PER_MIN = 1.0 / 90.0    # 0 -> 80% 约 72 分钟
CHARGE_CV_PER_MIN = 1.0 / 240.0   # 80% -> 100% 慢下来

CUTOFF_SOC = 0.03    # 低于该值视为断电（保护截止电压）
FULL_SOC = 0.999


def drain_rate(mic: Microphone, state: MicState) -> float:
    return DRAIN_PER_MIN[state] * mic.drain_scale / max(0.3, mic.battery.health)


def drain_one_minute(mic: Microphone, state: MicState) -> None:
    mic.battery.soc = max(0.0, mic.battery.soc - drain_rate(mic, state))


def charge_one_minute(b: Battery) -> None:
    rate = CHARGE_CC_PER_MIN if b.soc < 0.8 else CHARGE_CV_PER_MIN
    b.soc = min(FULL_SOC, b.soc + rate)


def time_to_empty(mic: Microphone, start: int, horizon: int,
                  soc: float | None = None) -> float | None:
    """预计耗尽时间。

    从 start 分钟开始，严格按未来使用计划逐分钟积分，返回 soc 跌到
    CUTOFF_SOC 的分钟数（可为小数）；horizon 内不会断电则返回 None。

    soc 为 start 时刻的（假设）电量；默认取电池当前电量——因此调用方
    若传入未来的 start，必须先用 :func:`predict_soc` 推出该时刻电量再
    传入，否则会漏掉 start 之前的耗电、高估续航。
    """
    if soc is None:
        soc = mic.battery.soc
    if soc <= CUTOFF_SOC:
        # 起始时刻已经低于保护截止电压：TTE 为 0（不返回负值/过去时刻）
        return float(start)
    for t in range(start, horizon):
        rate = drain_rate(mic, mic.state_at(t))
        soc -= rate
        if soc <= CUTOFF_SOC:
            # 线性回退到越过阈值的精确时刻
            overshoot = CUTOFF_SOC - soc
            return t - min(1.0, overshoot / rate)
    return None


def predict_soc(mic: Microphone, start: int, target: int) -> float:
    """不改变现场，推演 mic 在 target 分钟时的电量（中间无换电）。"""
    soc = mic.battery.soc
    for t in range(start, target):
        soc = max(0.0, soc - drain_rate(mic, mic.state_at(t)))
    return soc
