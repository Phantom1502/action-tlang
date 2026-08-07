from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

@dataclass(frozen=True)
class BaseConfig:
    bin_min: int
    bin_max: int
    n_bins: int
    zone_width_min_bins: int
    zone_width_max_bins: int
    zone_score_weight: float
    zone_extend_multiplier: float
    zone_last_n_touch: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int
    digit_pad: int
    rr_min: int
    rr_max: int
    
@dataclass(frozen=True)
class WindowConfig:
    input_candles: int      # 100 o v2
    outcome_horizon: int    # 100 o v2
    window_size: int        # PHAI = input_candles + outcome_horizon

    def __post_init__(self) -> None:
        if self.window_size != self.input_candles + self.outcome_horizon:
            raise ValueError(
                "WindowConfig.window_size phai bang input_candles + outcome_horizon "
                f"(nhan duoc window_size={self.window_size}, "
                f"input_candles={self.input_candles}, "
                f"outcome_horizon={self.outcome_horizon})"
            )
            
@dataclass(frozen=True)
class AppConfig:
    base: BaseConfig
    window: WindowConfig