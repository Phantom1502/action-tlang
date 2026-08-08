from __future__ import annotations

from typing import List, Tuple
import random

from app.lang import (
    ProgramNode,
    ChartNode,
    CandleNode,
    ThinkNode,
    ZoneNode
)

class DataAugmenter:
    def __init__(self, rng: random.Random, n_bins: int):
        self.rng = rng
        self.n_bins = n_bins
        
    def augment_shift(
        self, 
        program: ProgramNode,
        n_augments: int = 1
    ) -> List[Tuple[str, ProgramNode]]:
        if program.chart is None or program.think is None:
            return []
        lows = [c.low for c in program.chart.candles]
        highs = [c.high for c in program.chart.candles]
        
        lows.append(program.think.zone.lower_bin)
        highs.append(program.think.zone.upper_bin)
        
        if program.future_bins is not None:
            lows.extend([c.low for c in program.future_bins])
            highs.extend([c.high for c in program.future_bins])
        min_low, max_high = min(lows), max(highs)
        
        shift_min = -min_low
        shift_max = (self.n_bins - 1) - max_high
        if shift_min > shift_max:
            return []

        choices = [d for d in range(shift_min, shift_max + 1) if d != 0]
        if not choices:
            return []

        n_augments = min(n_augments, len(choices))
        samples = self.rng.sample(choices, n_augments)
        
        results = []
        for delta in samples:
            new_chart = ChartNode(candles=[CandleNode(c.open + delta, c.high + delta, c.low + delta, c.close + delta) for c in program.chart.candles])
            new_think = ThinkNode(
                trend=program.think.trend, 
                current_price_bin=program.think.current_price_bin + delta, 
                zone=ZoneNode(program.think.zone.direction, program.think.zone.lower_bin + delta, program.think.zone.upper_bin + delta)
            )
            results.append((
                f"aug_{delta}",
                ProgramNode(
                    chart=new_chart, 
                    think=new_think,
                    future_bins=[CandleNode(c.open + delta, c.high + delta, c.low + delta, c.close + delta) for c in program.future_bins] 
                    if program.future_bins is not None else None
                )
            ))
        return results