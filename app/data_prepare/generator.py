from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from app.lang import (
    ASTVisitor,
    Parser,
    ParseResult,
    SemanticChecker,
    ProgramNode,
    ActionNode,
    CandleNode,
    ChartNode,
    ThinkNode,
    ZoneNode
)

LEAF_RECIPES = {
    "support": {
        "actions": ["BUY", "HOLD"],
        "probs": [0.8, 0.2]
    },
    "resistance": {
        "actions": ["SELL", "HOLD"],
        "probs": [0.8, 0.2]
    }
}

def _pick_sl_rr(
    rng: random.Random, 
    action_type: str, 
    current_price: int, 
    zone: ZoneNode,
    sl_min_dist: int,
    sl_max_dist: int,
    bin_min: int,
    bin_max: int,
    rr_min: int,
    rr_max: int
) -> Optional[Tuple[int, int]]:
    # pick sl
    dist_candidates = list(range(sl_min_dist, sl_max_dist + 1))
    rng.shuffle(dist_candidates)
    for dist in dist_candidates:
        if action_type == "BUY":
            sl = current_price - dist
            if not (bin_min <= sl < zone.lower_bin): 
                continue
        else:
            sl = current_price + dist
            if not (zone.upper_bin < sl <= bin_max):
                continue

        # pick rr
        rr_candidates = list(range(rr_min, rr_max + 1))
        rng.shuffle(rr_candidates)
        rr = rng.choice(rr_candidates)
        return sl, rr

    return None

@dataclass
class GeneratedSample:
    prompt: str
    completion: str
    program: ProgramNode
    leaf_recipe: str

class ActionGenerator:
    def __init__(
        self,
        cfg: AppConfig,
        seed: Optional[int] = None
    ):
        self.cfg = cfg
        self._random = random.Random(seed)
        self._ast_visitor = ASTVisitor(digit_pad=cfg.base.digit_pad)
    
    def generate_one(
        self,
        program: ProgramNode,
        max_attempts: int = 30,
    ) -> Optional[GeneratedSample]:
        current_price = program.chart.current_price
        
        if program.think.zone is None:
            # re-verify mặc dù thực tế, với filter data, điểm này phải mặc định đúng 
            return None
        
        for _ in range(max_attempts):
            recipe = LEAF_RECIPES[program.think.zone.direction]
            action_type = np.random.choice(recipe["actions"], p=recipe["probs"])
            action = ActionNode(action_type=action_type)
            
            if action_type in ("BUY", "SELL"):
                sl_rr = _pick_sl_rr(
                    self._random, 
                    action_type, 
                    current_price,
                    think.zone,
                    self.cfg.base.sl_min_dist_bins,
                    self.cfg.base.sl_max_dist_bins,
                    self.cfg.base.bin_min,
                    self.cfg.base.bin_max,
                    self.cfg.base.rr_min,
                    self.cfg.base.rr_max
                )
                if sl_rr is None:
                    continue
                action.sl, action.rr = sl_rr
                
            prompt = self._ast_visitor.build_action_prompt(program.chart.candles, program.think)
            completion = self._ast_visitor.build_action_completion(action)
            
            parse_result: ParseResult = Parser.from_text(self.cfg, prompt + " " + completion).parse()
            if not parse_result.is_well_formed():
                continue
            
            semantic_checker = SemanticChecker(
                zone_width_min_bins=cfg.base.zone_width_min_bins,
                zone_width_max_bins=cfg.base.zone_width_max_bins,
                sl_min_dist_bins=cfg.base.sl_min_dist_bins,
                sl_max_dist_bins=cfg.base.sl_max_dist_bins,
                zone_extend_multiplier=cfg.base.zone_extend_multiplier,
                last_n_touch=cfg.base.zone_last_n_touch
            )
            semantic_result = semantic_checker.check(parse_result.ast)
            if not semantic_result.passed:
                continue
            
            program.action = action
            
            leaf_name = f"{think.zone.direction}|{action_type}"
            return GeneratedSample(prompt, completion, program, leaf_name)
        
        return None
    
    def generate_dataset(
        self, 
        programs: List[ProgramNode],
        samples_per_chart: int = 4,
        max_attempts: int = 30,
    ) -> List[GeneratedSample]:
        samples: List[GeneratedSample] = []
        for program in programs:
            for _ in range(samples_per_chart):
                sample = self.generate_one(program, max_attempts=max_attempts)
                if sample is not None:
                    samples.append(sample)
        return samples
    
if __name__ == "__main__":
    def make_chart(closes) -> List[CandleNode]:
        candles = [CandleNode(c, c + 5, c - 5, c) for c in closes] 
        return candles

    from app.config import load_config, AppConfig
    cfg: AppConfig = load_config("configs")
    
    closes = [500, 505, 503, 507, 510]
    futures_close = [515, 504, 505, 510, 527]
    chart = ChartNode(candles=make_chart(closes))    
    think = ThinkNode(
        trend="UP", 
        current_price_bin=closes[-1], 
        zone=ZoneNode(
            direction="support", # support |resistance
            lower_bin=500, 
            upper_bin=505
        )
    )
    program = ProgramNode(chart=chart, think=think, future_bins=make_chart(futures_close))
    action_generator = ActionGenerator(cfg)
    samples = action_generator.generate_dataset([program], samples_per_chart=10)
    
    
    for sample in samples:
        print(sample.completion)
        print(sample.leaf_recipe)