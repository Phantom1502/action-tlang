from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CandleNode:
    open:   int
    high:   int
    low:    int
    close:  int


@dataclass
class ChartNode:
    candles: List[CandleNode] = field(default_factory=list)

    @property
    def current_price(self) -> int:
        return self.candles[-1].close

@dataclass
class ZoneNode:
    direction: str        # "support" | "resistance"
    lower_bin: int
    upper_bin: int

@dataclass
class ThinkNode:
    trend: Optional[str] = None                  # "UP" | "DOWN" | "RANGE"
    current_price_bin: Optional[int] = None      # BẮT BUỘC theo spec — luôn phải có mặt
    zone: Optional[ZoneNode] = None

@dataclass
class ActionNode:
    action_type: Optional[str] = None  # BUY | SELL | HOLD
    sl: Optional[int] = None
    rr: Optional[int] = None           # risk luôn chuẩn hoá = 1, rr là reward-multiple duy nhất

    @property
    def is_hold(self) -> bool:
        return self.action_type == "HOLD"

@dataclass
class ProgramNode:
    chart: Optional[ChartNode] = None
    think: Optional[ThinkNode] = None
    action: Optional[ActionNode] = None
    future_bins: Optional[List[CandleNode]] = None
