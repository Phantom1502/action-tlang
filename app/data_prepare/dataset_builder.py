from __future__ import annotations

import numpy as np
import random
from typing import List, Tuple, Literal, Optional

from app.config import load_config, AppConfig, get_scale
from .common import gen_action
from tlang import (
    ChartCodec,
    ZoneDirection,
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

def contruct_record(
    chart: ChartNode,
    think: ThinkNode,
    action: ActionNode,
    digit_pad: int
)-> Tuple[str, str]:
    ast_visitor = ASTVisitor(digit_pad=digit_pad)
    prompt = ast_visitor.render_chart_block(chart.candles)
    parts = []
    parts.append(ast_visitor.visit_think(think))
    parts.append(ast_visitor.visit_action(action))
    completion = " ".join(parts)
    return prompt, completion

def build_pretrain_dataset(
    cfg: AppConfig,
    input_dir: str,
    output_dir: str,
    train_file: str = "train.parquet",
    n_samples: int = 3,
    trend_threshhold: float = 0.6,
    hold_threshhold: float = 0.3,
    swing_window: int = 5,
    seed: int = 42
):
    from datasets import load_dataset
    import os
    from collections import Counter
    
    data_files = {
        "train": f"{input_dir}/{train_file}",
        "val": f"{input_dir}/window_200_val.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    counter = Counter()
    zone_counter = Counter()
    def preprocess_for_llm(batch):
        prompts = []
        completions = []
        symbols = []
        
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
            
            zone_noise = (cfg.tlang.zone_range[1] - cfg.tlang.zone_range[0]) // 2
            zone_nodes = []
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
                for future_idx, lower_bin, upper_bin, width in zones:
                    zone = ZoneNode(zone_direction, lower_bin, upper_bin)
                    zone_counter[f"BEFORE_FILLTER_{zone_direction.value}"] += 1
                    # score tính trên zone tối ưu ko noise
                    score = zone_score(
                        zone, 
                        future_candles, 
                        cfg.base.rr_min, 
                        cfg.base.rr_max,
                        cfg.tlang.n_bins-1
                    ) * cfg.base.zone_score_weight
                    if score > hold_threshhold:
                        zone_nodes.append((score, zone))
                        zone_counter[zone_direction.value] += 1
                        # thêm noise để đa dạng zone
                        for _ in range(n_samples):
                            if zone_noise > 0:
                                lower_bin = max(0, zone.lower_bin - random.randint(0, zone_noise))
                                upper_bin = min(cfg.tlang.n_bins - 1, zone.upper_bin + random.randint(0, zone_noise))
                            
                            noise_zone = ZoneNode(zone_direction, lower_bin, upper_bin)
                            zone_nodes.append((score, noise_zone))
                
            records = []
            for score, zone in zone_nodes:
                action = gen_action(
                    cfg.tlang,
                    zone,
                    chart.current_price,
                    future_candles,
                    cfg.base.rr_min,
                    cfg.base.rr_max,
                    0,
                )
                trend = TrendType.RANGE
                if score > trend_threshhold and zone.direction == ZoneDirection.support:
                    trend = TrendType.UP
                elif score > trend_threshhold and zone.direction == ZoneDirection.resistance:
                    trend = TrendType.DOWN
                think = ThinkNode(trend=trend, current_price_bin=chart.current_price, zone=zone)
                prompt, completion = contruct_record(chart, think, action, cfg.tlang.digit_pad)
                counter[f"{trend.value}_{zone.direction.value}_{action.action_type.value}"] += 1
                
                records.append((prompt, completion))
                
                noise_action = gen_action(
                    cfg.tlang,
                    zone,
                    chart.current_price,
                    future_candles,
                    cfg.base.rr_min,
                    cfg.base.rr_max,
                    cfg.tlang.sl_range[1] - cfg.tlang.sl_range[0],
                )
                noise_think = ThinkNode(trend=trend, current_price_bin=chart.current_price, zone=zone)
                noise_prompt, noise_completion = contruct_record(chart, noise_think, noise_action, cfg.tlang.digit_pad)
                counter[f"{trend.value}_{zone.direction.value}_{noise_action.action_type.value}"] += 1
                records.append((noise_prompt, noise_completion))
                
            if len(records) == 0:
                records.append(
                    contruct_record(
                        chart, 
                        ThinkNode(
                            trend=TrendType.RANGE, 
                            current_price_bin=chart.current_price, 
                            zone=None
                        ), 
                        ActionNode(
                            action_type=ActionType.HOLD, 
                            sl=None, 
                            rr=None
                        ), 
                        cfg.tlang.digit_pad
                    )
                )
                zone_counter["NO_ZONE"] += 1
                counter[f"{TrendType.RANGE.value}_NO_ZONE_{ActionType.HOLD.value}"] += 1
                
            for record in records:
                prompts.append(record[0])
                completions.append(record[1])
                symbols.append(symbol)
                
        print(f"Zone counter: total {zone_counter.total()} records, {zone_counter}")
        print(f"Counter: total {counter.total()} records, {counter}")
                
        # Trả về các cột mới cho Dataset LLM
        return {
            "prompt": prompts,
            "completion": completions,
            "symbol": symbols,
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
    
    llm_dataset["train"].shuffle(seed=seed).to_parquet(f"{output_dir}/train_pretrain.parquet")
    llm_dataset["val"].to_parquet(f"{output_dir}/val_pretrain.parquet") 
    
if __name__ == "__main__":
    cfg: AppConfig = load_config("configs")
    build_pretrain_dataset(cfg, "data/dataset", "data/dataset")