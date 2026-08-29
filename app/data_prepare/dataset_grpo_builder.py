 
from __future__ import annotations

import numpy as np
import random
from typing import List, Optional, Tuple, Literal, Dict

from app.config import AppConfig, load_config, get_scale
from .common import gen_action
from tlang import (
    ChartCodec,
    ZoneDirection,
    ProgramNode,
    ChartNode,
    ThinkNode,
    ZoneNode,
    ActionNode,
    ActionType,
    TrendType,
    ASTVisitor,
    zone_score,
    find_truly_valid_zones,
)

from collections import Counter
    
def build_grpo_dataset(
    cfg: AppConfig,
    input_dir: str,
    output_dir: str,
    seed: Optional[int] = None,
    trend_threshhold: float = 0.6,
    hold_threshhold: float = 0.3,
    swing_window: int = 5,
):
    from datasets import load_dataset
    import os
    
    data_files = {
        "train": f"{input_dir}/window_200_train.parquet",
        "val": f"{input_dir}/window_200_val.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    counter = Counter()
    ast_visitor = ASTVisitor(digit_pad=cfg.tlang.digit_pad)
    def preprocess_for_llm(batch):
        symbols_records = []
        prompts_records = []
        future_bins_records = []
        trends_records = []
        zones_records = []
        zone_ranges_records = []
        actions_records = []
        sls_records = []
        rrs_records = []
        
        batch_size = len(batch["symbol"])
        
        # Duyệt qua các phần tử trong batch (chạy trong RAM của batch đó, cực nhẹ)
        for i in range(batch_size):
            symbol_tf = batch["symbol"][i]
            input_window = np.array(batch["input_window"][i], dtype=np.float32)
            future_window = np.array(batch["future_window"][i], dtype=np.float32)
            atr_100 = batch["atr_100"][i]
            
            symbol = symbol_tf.split("_")[0]
            timeframe = symbol_tf.split("_")[1]
            scale = get_scale(cfg, symbol, timeframe)
                        
            codec = ChartCodec(scale, cfg.tlang.n_bins)
            input_candles, anchor_open = codec._encode_input(input_window, atr_100)
            future_candles = codec._encode_future(future_window, anchor_open, atr_100)
            chart: ChartNode = ChartNode(candles=input_candles)
                   
            results: List[Tuple[int, float, ProgramNode]] = []     
            for zone_direction in [ZoneDirection.support, ZoneDirection.resistance]:
                zones = find_truly_valid_zones(
                    input_candles, 
                    cfg.tlang.last_n_touch, 
                    future_candles, 
                    zone_direction, 
                    swing_window=swing_window, 
                    zone_width=cfg.tlang.zone_range[0],
                    max_bin=cfg.tlang.n_bins
                )
                zone_qualitys: Dict[float, Tuple[int, float, ProgramNode]] = {}
                for future_idx, lower_bin, upper_bin, width in zones:
                    zone = ZoneNode(zone_direction, lower_bin, upper_bin)
                    action = gen_action(
                        cfg.tlang,
                        zone,
                        chart.current_price,
                        future_candles,
                        cfg.base.rr_min,
                        cfg.base.rr_max,
                        0,
                    )
                    score = zone_score(zone, future_candles, cfg.base.rr_min, cfg.base.rr_max) * cfg.base.zone_score_weight
                    if score > trend_threshhold:
                        if zone_direction == ZoneDirection.support:
                            trend = TrendType.UP
                        else:
                            trend = TrendType.DOWN
                        
                        program: ProgramNode = ProgramNode(
                            chart=chart,
                            think=ThinkNode(
                                trend=trend,
                                current_price_bin=chart.current_price,
                                zone=ZoneNode(zone_direction, lower_bin, upper_bin)
                            ),
                            action=action
                        )
                        zone_qualitys[score] = (future_idx, score, program)
                    elif score > hold_threshhold:
                        program: ProgramNode = ProgramNode(
                            chart=chart,
                            think=ThinkNode(
                                trend=TrendType.RANGE,
                                current_price_bin=chart.current_price,
                                zone=ZoneNode(zone_direction, lower_bin, upper_bin)
                            ),
                            action=action
                        )
                        zone_qualitys[score] = (future_idx, score, program)
                
                # giữ lại zone tốt nhất cho zone type này
                if len(zone_qualitys) > 0:
                    results.append(zone_qualitys[max(zone_qualitys.keys())]) 
                
            best_program = None
            if len(results) > 0:
                if len(results) == 1: # tìm được 1 zone duy nhất
                    idx, best_quality, best_program = results[0]
                elif len(results) > 1:
                    # find nearest zone by min idx
                    min_idx, best_quality, best_program = min(results, key=lambda x: (x[0], -x[1]))

            if best_program is None:
                best_program: ProgramNode = ProgramNode(
                    chart=chart,
                    think=ThinkNode(
                        trend=TrendType.RANGE,
                        current_price_bin=chart.current_price,
                    ),
                    action=ActionNode(
                        action_type=ActionType.HOLD
                    )
                )
                counter[f"{best_program.think.trend.value}_NOZONE"] += 1
            else:
                counter[f"{best_program.think.trend.value}_{best_program.think.zone.direction.value}"] += 1
            
            prompt = ast_visitor.render_chart_block(best_program.chart.candles)
            prompts_records.append(prompt)
            future_bins_records.append([[int(candle.open), int(candle.high), int(candle.low), int(candle.close)] for candle in future_candles])
            trends_records.append(best_program.think.trend.value)
            zones_records.append(best_program.think.zone_type.value)
            zone_ranges_records.append([int(best_program.think.zone.lower_bin), int(best_program.think.zone.upper_bin)] if best_program.think.zone is not None else [0, 0])
            actions_records.append(best_program.action.action_type.value)
            sls_records.append(int(best_program.action.sl) if best_program.action.sl is not None else 0)
            rrs_records.append(int(best_program.action.rr) if best_program.action.rr is not None else 0)
            symbols_records.append(symbol_tf)
               
        print(counter) 
        # Trả về các cột mới cho Dataset LLM
        return {
            "prompt": prompts_records,
            "future_bins": future_bins_records,
            "trends": trends_records,
            "zones": zones_records,
            "zone_ranges": zone_ranges_records,
            "actions": actions_records,
            "sls": sls_records,
            "rrs": rrs_records,
            "symbol": symbols_records,
        }
        
    llm_dataset = dataset.map(
        preprocess_for_llm,
        batched=True,
        batch_size=2000, # Mỗi lần nạp 2000 dòng vào RAM để parse
        num_proc=os.cpu_count(),      # Số lượng nhân CPU chạy song song
        remove_columns=dataset["train"].column_names # Xóa các cột gốc (id, type, score...) để thu gọn dataset
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    llm_dataset["train"].shuffle(seed=seed).to_parquet(f"{output_dir}/train_grpo.parquet")
    llm_dataset["val"].to_parquet(f"{output_dir}/val_grpo.parquet") 
    
if __name__ == "__main__":
    cfg: AppConfig = load_config("configs")
    build_grpo_dataset(cfg, "data/dataset", "data/dataset")