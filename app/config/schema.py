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
class ModelPreset:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int

@dataclass(frozen=True)
class ModelsConfig:
    vocab_size: int
    max_position_embeddings: int
    presets: Dict[str, ModelPreset] = field(default_factory=dict)  # key: "tiny"/"small"/"base"/"large"
   
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    logging_steps: int
    save_steps: int
    
    def __post_init__(self):
        # Validate & convert learning_rate
        try:
            lr_val = float(self.learning_rate)
            object.__setattr__(self, "learning_rate", lr_val)
        except (ValueError, TypeError):
            raise TypeError(f"[{self.phase}] learning_rate không thể convert sang float: {self.learning_rate!r}")

        # Validate các field int
        int_fields = ["batch_size", "gradient_accumulation_steps", "warmup_steps", "max_steps", "logging_steps", "save_steps"]
        for field in int_fields:
            val = getattr(self, field)
            try:
                object.__setattr__(self, field, int(val))
            except (ValueError, TypeError):
                raise TypeError(f"[{self.phase}] {field} không thể convert sang int: {val!r}")
      
@dataclass(frozen=True)
class AppConfig:
    base: BaseConfig
    window: WindowConfig
    models: ModelsConfig
    training: Dict[str, TrainingConfig]