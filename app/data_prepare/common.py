from __future__ import annotations

import random
from typing import List
from tlang import (
    TLangConfig,
    ZoneDirection,
    CandleNode,
    ZoneNode,
    ActionNode,
    ActionType,
    find_best_rr,
    find_entry_touch
)

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
            # đoạn này thừa, vì entry = current price nghĩa là đã vào
            # find best RR for buy action with entry = current_price
            #first_touch = find_entry_touch(current_price, ActionType.BUY, future_candles)
            #if first_touch is None:
            #    return ActionNode(action_type, None, None)
            rr = find_best_rr(ActionType.BUY, current_price, sl, future_candles, rr_min, rr_max)
            action_type = ActionType.BUY
        elif current_price > zone.upper_bin: # giá trên zone, entry là upper bin
            sl_range = max(zone.upper_bin - zone.lower_bin + 1, tlang_cfg.sl_range[0])
            sl = zone.upper_bin - sl_range
            if sl < 0: # sl out of range
                return ActionNode(action_type, None, None)
            sl = max(sl - random.randint(0, noise), 0)
            # find best RR for buy action with entry = upper bin
            first_touch = find_entry_touch(zone.upper_bin, ActionType.BUY, future_candles)
            if first_touch is None:
                return ActionNode(action_type, None, None)
            rr = find_best_rr(ActionType.BUY, zone.upper_bin, sl, future_candles[first_touch:], rr_min, rr_max)
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
            #first_touch = find_entry_touch(current_price, ActionType.SELL, future_candles)
            #if first_touch is None:
            #    return ActionNode(action_type, None, None)
            rr = find_best_rr(ActionType.SELL, current_price, sl, future_candles, rr_min, rr_max)
            action_type = ActionType.SELL
        elif current_price < zone.lower_bin: # giá dưới zone, entry là lower bin
            sl_range = max(zone.upper_bin - zone.lower_bin + 1, tlang_cfg.sl_range[0])
            sl = zone.lower_bin + sl_range
            if sl >= tlang_cfg.n_bins: # sl out of range
                return ActionNode(action_type, None, None)
            sl = min(sl + random.randint(0, noise), tlang_cfg.n_bins - 1)
            # find best RR for sell action with entry = lower bin
            first_touch = find_entry_touch(zone.lower_bin, ActionType.SELL, future_candles)
            if first_touch is None:
                return ActionNode(action_type, None, None)
            rr = find_best_rr(ActionType.SELL, zone.lower_bin, sl, future_candles[first_touch:], rr_min, rr_max)
            action_type = ActionType.SELL
        else: # giá trên zone, zone không còn hợp lệ
            return ActionNode(action_type, None, None)

    if rr < rr_min:   # chưa đạt nổi cả mức RR thấp nhất trước khi SL chạm -> không phải setup đáng ghi nhận
        return ActionNode(ActionType.HOLD, None, None)
    return ActionNode(action_type=action_type, sl=sl, rr=rr)