from __future__ import annotations

import numpy as np
import random
from typing import List, Tuple, Literal

from app.config import load_config, AppConfig, get_scale
from app.training.reward import TLangReward, derive_target
from tlang import (
    TLangConfig,
    ChartCodec,
    ZoneDirection,
    ProgramNode,
    ChartNode,
    CandleNode,
    ThinkNode,
    ZoneNode,
    ActionNode,
    ActionType,
    TrendType,
    ASTVisitor,
    plot_zones,
    plot_program
)

def find_truly_valid_zones(
    input_candles: List[CandleNode],
    last_n: int,
    future_candles: List[CandleNode],
    mode: Literal[ZoneDirection.support, ZoneDirection.resistance] = ZoneDirection.support,
    swing_window: int = 2,
    zone_width: int = 50,
    max_bin: int = 2047
) -> List[Tuple[int, int, int]]:
    last_n_candles = input_candles[-last_n:]
    candles = last_n_candles + future_candles
    n = len(candles)
    
    valid_input_min = min(c.low for c in input_candles)
    valid_input_max = max(c.high for c in input_candles)
    valid_zone_ranges = [valid_input_min, valid_input_max]
    
    valid_swings = []
    if n <= last_n:
        return valid_swings
    
    is_support = (mode == ZoneDirection.support)
    for i in range(last_n, n):
        # 1. KIỂM TRA ĐIỀU KIỆN SWING HIGH / SWING LOW
        # Đảm bảo đủ số nến swing_window ở hai bên
        left_start = max(0, i - swing_window)
        right_end = min(n - 1, i + swing_window)

        if is_support:
            current_val = candles[i].low
            # Là Swing Low nếu giá Low hiện tại <= tất cả các nến trong cửa sổ xung quanh
            is_swing = all(current_val <= candles[j].low for j in range(left_start, right_end + 1) if j != i)
        else:
            current_val = candles[i].high
            # Là Swing High nếu giá High hiện tại >= tất cả các nến trong cửa sổ xung quanh
            is_swing = all(current_val >= candles[j].high for j in range(left_start, right_end + 1) if j != i)

        if not is_swing:
            continue
        
        if len(valid_swings) == 0:
            valid_swings.append((i, current_val))
        else:
            if is_support:
                min_swing = valid_swings[-1][1]
                if current_val < min_swing:
                    valid_swings.append((i, current_val))
            else:
                max_swing = valid_swings[-1][1]
                if current_val > max_swing:
                    valid_swings.append((i, current_val))

    # group all nearby swings into zones
    valid_zones = []
    for idx, swing in valid_swings:
        if len(valid_zones) == 0:
            valid_zones.append((idx, swing, swing))
        else:
            id, slow, shigh = valid_zones[-1]
            if is_support:
                if shigh - swing <= zone_width:
                    valid_zones[-1] = (id, swing, shigh)
                else:
                    valid_zones.append((idx, swing, swing))
            else:
                if swing - slow <= zone_width:
                    valid_zones[-1] = (id, slow, swing)
                else:
                    valid_zones.append((idx, swing, swing))

    # extend valid_zones to zone_width
    results = []
    for id, slow, shigh in valid_zones:
        if is_support:
            if shigh - slow <= zone_width:
                remain = zone_width - (shigh - slow)
                lower_bin = max(0, slow - remain // 2)
                upper_bin = lower_bin + zone_width
                    
                if upper_bin >= valid_zone_ranges[0]:
                    results.append((id, lower_bin, upper_bin, upper_bin - lower_bin))
        else:
            if shigh - slow <= zone_width:
                remain = zone_width - (shigh - slow)
                upper_bin = min(max_bin, shigh + remain // 2)
                lower_bin = upper_bin - zone_width
                    
                if lower_bin <= valid_zone_ranges[1]:
                    results.append((id, lower_bin, upper_bin, upper_bin - lower_bin))

    return results

def find_best_rr(
    action_type: ActionType,
    entry_price: int,
    sl: int,
    future_candles: List[CandleNode],
    rr_min: int,
    rr_max: int
)-> int:
    best_rr = 0
    for rr in range(rr_min, rr_max + 1):
        if action_type == ActionType.BUY:
            target = derive_target(entry_price, sl, rr, "long")
            for candle in future_candles:
                hit_sl = candle.low <= sl
                if hit_sl:
                    return rr
                hit_target = candle.high >= target
                if hit_target:
                    best_rr = rr
                    continue
        else:
            target = derive_target(entry_price, sl, rr, "short")
            for candle in future_candles:
                hit_sl = candle.high >= sl
                if hit_sl:
                    return rr
                hit_target = candle.low <= target
                if hit_target:
                    best_rr = rr
                    continue
    return best_rr

def gen_action(
    tlang_cfg: TLangConfig,
    zone: ZoneNode,
    current_price: int,
    future_candles: List[CandleNode],
    rr_min: int,
    rr_max: int,
    noise: int
):
    action_type = ActionType.HOLD
    sl = None
    rr = None
    if zone.direction == ZoneDirection.support:
        if zone.lower_bin <= current_price <= zone.upper_bin: # in zone
            sl_range = max(current_price - zone.lower_bin + 1, tlang_cfg.sl_range[0])
            sl = current_price - sl_range
            if sl < 0: # sl out of range
                return ActionNode(action_type, None, None)
            sl = max(sl - random.randint(0, noise), 0)
            # find best RR for buy action with entry = current_price
            rr = find_best_rr(ActionType.BUY, current_price, sl, future_candles, rr_min, rr_max)
            action_type = ActionType.BUY
        elif current_price > zone.upper_bin: # giá trên zone, entry là upper bin
            sl_range = max(zone.upper_bin - zone.lower_bin + 1, tlang_cfg.sl_range[0])
            sl = zone.upper_bin - sl_range
            if sl < 0: # sl out of range
                return ActionNode(action_type, None, None)
            sl = max(sl - random.randint(0, noise), 0)
            # find best RR for buy action with entry = upper bin
            rr = find_best_rr(ActionType.BUY, zone.upper_bin, sl, future_candles, rr_min, rr_max)
            action_type = ActionType.BUY
        else: # giá dưới zone, zone không còn hợp lệ
            return ActionNode(action_type, None, None)
    else: # zone.direction == ZoneDirection.resistance
        if zone.lower_bin <= current_price <= zone.upper_bin: # in zone
            sl_range = max(zone.upper_bin - current_price + 1, tlang_cfg.sl_range[0])
            sl = current_price + sl_range
            if sl >= tlang_cfg.n_bins: # sl out of range
                return ActionNode(action_type, None, None)
            sl = min(sl + random.randint(0, noise), tlang_cfg.n_bins - 1)
            # find best RR for sell action with entry = current_price
            rr = find_best_rr(ActionType.SELL, current_price, sl, future_candles, rr_min, rr_max)
            action_type = ActionType.SELL
        elif current_price < zone.lower_bin: # giá dưới zone, entry là lower bin
            sl_range = max(zone.upper_bin - zone.lower_bin + 1, tlang_cfg.sl_range[0])
            sl = zone.lower_bin + sl_range
            if sl >= tlang_cfg.n_bins: # sl out of range
                return ActionNode(action_type, None, None)
            sl = min(sl + random.randint(0, noise), tlang_cfg.n_bins - 1)
            # find best RR for sell action with entry = lower bin
            rr = find_best_rr(ActionType.SELL, zone.lower_bin, sl, future_candles, rr_min, rr_max)
            action_type = ActionType.SELL
        else: # giá trên zone, zone không còn hợp lệ
            return ActionNode(action_type, None, None)

    return ActionNode(action_type=action_type, sl=sl, rr=rr)

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
    
    data_files = {
        "train": f"{input_dir}/{train_file}",
        "val": f"{input_dir}/window_200_val.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    
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
                        
            reward = TLangReward(cfg)
            
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
                    
                    # score tính trên zone tối ưu ko noise
                    score = reward.zone_score(zone, future_candles).zone_quality
                    print(f"Zone: {zone.direction} {zone.lower_bin} {zone.upper_bin} {score}")
                    if score > hold_threshhold:
                        zone_nodes.append((score, zone))
                        
                        # thêm noise để đa dạng zone
                        for _ in range(n_samples):
                            if zone_noise > 0:
                                lower_bin = max(0, zone.lower_bin - random.randint(0, zone_noise))
                                upper_bin = min(cfg.tlang.n_bins, zone.upper_bin + random.randint(0, zone_noise))
                            
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
                noise_think = ThinkNode(trend=trend, current_price_bin=chart.current_price, zone=noise_zone)
                noise_prompt, noise_completion = contruct_record(chart, noise_think, noise_action, cfg.tlang.digit_pad)
                
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
                
            for record in records:
                prompts.append(record[0])
                completions.append(record[1])
                symbols.append(symbol)
                
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