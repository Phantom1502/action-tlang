from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any
from tlang import TLangConfig

@dataclass(frozen=True)
class BaseConfig:
    trade_fee_bins: int
    zone_score_weight: float
    entry_score_weight: float
    outcome_score_weight: float
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
class ScaleEntry:
    symbol: str
    timeframe: str
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"ScaleEntry.scale phai > 0 (nhan duoc {self.scale})")
        
@dataclass(frozen=True)
class EntropyConfig:
    """Cau hinh buff cho DUNG 1 action_type (khong gop nhom nhu v1)."""
    floor: float
    ema_alpha: float
    kp: float
    kd: float
    bonus_step_max: float
    bonus_cap: float
        
@dataclass(frozen=True)
class RoundConfig:
    round_id: str
    alpha: float
    kp: float
    kd: float
    step_max: int
    entropys: Dict[str, EntropyConfig]
     
@dataclass(frozen=True)
class AppConfig:
    base: BaseConfig
    window: WindowConfig
    scales: List[ScaleEntry]
    models: ModelsConfig
    training: Dict[str, TrainingConfig]
    rounds: Dict[str, RoundConfig]
    tlang: TLangConfig