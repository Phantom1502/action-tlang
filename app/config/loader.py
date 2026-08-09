import os
from typing import Any, Dict

import yaml

from app.config.schema import (
    AppConfig,
    BaseConfig,
    WindowConfig,
    ModelPreset,
    ModelsConfig,
    TrainingConfig,
    RoundConfig,
    ZoneBuffConfig,
    ActionBuffConfig,
    GroupBuffState,
)

# Cac file bat buoc phai co truc tiep trong config_dir (khong ke rounds/, duoc xu ly rieng).
_REQUIRED_TOP_LEVEL_FILES = (
    "base.yaml",
    "window.yaml",
    "models.yaml",
    "training_defaults.yaml",
)
_ROUNDS_SUBDIR = "rounds"

def _read_yaml(path: str) -> Any:
    """Doc 1 file YAML, tra ve du lieu da parse (dict/list/...)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def _require_field(data: Dict[str, Any], key: str, source: str) -> Any:
    """Lay 1 field bat buoc tu dict; raise ValueError neu thieu (khong fallback am tham)."""
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"Thieu field bat buoc '{key}' trong {source}")
    return data[key]

def _build_base_config(data: Dict[str, Any], source: str) -> BaseConfig:
    return BaseConfig(
        bin_min=_require_field(data, "bin_min", source),
        bin_max=_require_field(data, "bin_max", source),
        n_bins=_require_field(data, "n_bins", source),
        zone_width_min_bins=_require_field(data, "zone_width_min_bins", source),
        zone_width_max_bins=_require_field(data, "zone_width_max_bins", source),
        zone_extend_multiplier=_require_field(data, "zone_extend_multiplier", source),
        zone_last_n_touch=_require_field(data, "zone_last_n_touch", source),
        sl_min_dist_bins=_require_field(data, "sl_min_dist_bins", source),
        sl_max_dist_bins=_require_field(data, "sl_max_dist_bins", source),
        trade_fee_bins=_require_field(data, "trade_fee_bins", source),
        entry_score_weight=_require_field(data, "entry_score_weight", source),
        outcome_score_weight=_require_field(data, "outcome_score_weight", source),
        digit_pad=_require_field(data, "digit_pad", source),
        rr_min=_require_field(data, "rr_min", source),
        rr_max=_require_field(data, "rr_max", source),
    )
    
def _build_window_config(data: Dict[str, Any], source: str) -> WindowConfig:
    return WindowConfig(
        input_candles=_require_field(data, "input_candles", source),
        outcome_horizon=_require_field(data, "outcome_horizon", source),
        window_size=_require_field(data, "window_size", source),
    )
    
def _build_models_config(data: Dict[str, Any], source: str) -> ModelsConfig:
    presets_raw = data.get("presets", {}) or {}
    presets = {}
    for name, preset_data in presets_raw.items():
        preset_source = f"{source}.presets.{name}"
        presets[name] = ModelPreset(
            hidden_size=_require_field(preset_data, "hidden_size", preset_source),
            num_hidden_layers=_require_field(preset_data, "num_hidden_layers", preset_source),
            num_attention_heads=_require_field(preset_data, "num_attention_heads", preset_source),
            num_key_value_heads=_require_field(preset_data, "num_key_value_heads", preset_source),
            intermediate_size=_require_field(preset_data, "intermediate_size", preset_source),
        )
    return ModelsConfig(
        vocab_size=_require_field(data, "vocab_size", source),
        max_position_embeddings=_require_field(data, "max_position_embeddings", source),
        presets=presets,
    )
    
def _build_training_config(data: Dict[str, Any], source: str) -> Dict[str, TrainingConfig]:
    """Parse dict config thành danh sách các đối tượng TrainingConfig."""
    if not isinstance(data, dict):
        raise ValueError(f"Cấu hình {source} phải là một dictionary!")

    return {
        k: TrainingConfig(
            batch_size=_require_field(v, "batch_size", source),
            gradient_accumulation_steps=_require_field(v, "gradient_accumulation_steps", source),
            learning_rate=_require_field(v, "learning_rate", source),
            warmup_steps=_require_field(v, "warmup_steps", source),
            max_steps=_require_field(v, "max_steps", source),
            logging_steps=_require_field(v, "logging_steps", source),
            save_steps=_require_field(v, "save_steps", source),
        )
        for k, v in data.items()
    }
    
def _build_round_config(data: Dict[str, Any], source: str) -> RoundConfig:
    support_zone_buffs = _require_field(data, "support", source)
    sup_actions = {
        k: ActionBuffConfig(
            buff_min=_require_field(v, "buff_min", source),
            buff_max=_require_field(v, "buff_max", source),
            buff_init=_require_field(v, "buff_init", source),
            target_ratio=_require_field(v, "target_ratio", source),
        )
        for k, v in support_zone_buffs.items()
    }
    support_buff: ZoneBuffConfig = ZoneBuffConfig(action_buffs=sup_actions)
    
    resistance_zone_buffs = _require_field(data, "resistance", source)
    res_actions = {
        k: ActionBuffConfig(
            buff_min=_require_field(v, "buff_min", source),
            buff_max=_require_field(v, "buff_max", source),
            buff_init=_require_field(v, "buff_init", source),
            target_ratio=_require_field(v, "target_ratio", source),
        )
        for k, v in resistance_zone_buffs.items()
    }
    resistance_buff: ZoneBuffConfig = ZoneBuffConfig(action_buffs=res_actions)
    
    return RoundConfig(
        round_id=_require_field(data, "round_id", source),
        alpha=_require_field(data, "alpha", source),
        kp=_require_field(data, "kp", source),
        kd=_require_field(data, "kd", source),
        step_max=_require_field(data, "step_max", source),
        zone_buffs={
            "support": support_buff,
            "resistance": resistance_buff,
        },
    )

def get_buff_group(round_config: RoundConfig, group_name: str, action_name: str) -> GroupBuffState:
    """
    Pre-condition: config da load thanh cong.
    Post-condition: tra ve dung GroupBuffState khop group_name.
    Raises: KeyError neu group_name khong ton tai.
    """
    if group_name not in round_config.zone_buffs:
        raise KeyError(f"Khong tim thay ZoneBuffConfig cho group_name={group_name!r}")
    if action_name not in round_config.zone_buffs[group_name].action_buffs:
        raise KeyError(f"Khong tim thay ActionBuffConfig cho action_name={action_name!r}")
    return GroupBuffState(
        ema_ratio=round_config.zone_buffs[group_name].action_buffs[action_name].target_ratio,
        buff=round_config.zone_buffs[group_name].action_buffs[action_name].buff_init,
        prev_error=0.0
    )
    
def load_config(config_dir: str = "./config") -> AppConfig:
    """
    Pre-condition: config_dir ton tai, chua du file bat buoc (base.yaml, window.yaml,
        scales.yaml, models.yaml, training_defaults.yaml, datagen_v2.yaml, rounds/*.yaml).
    Post-condition: tra ve dung 1 AppConfig da validate (moi __post_init__ cua
        schema.py deu pass).
    Raises:
        FileNotFoundError -- thieu 1 trong cac file bat buoc.
        ValueError -- 1 file co field bat buoc bi thieu, hoac gia tri vi pham
            invariant dinh nghia trong app/config/schema.py.
    Side-effect: chi doc file, KHONG ghi, KHONG mutate global state nao.
    """
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"config_dir khong ton tai: {config_dir}")

    paths = {}
    for filename in _REQUIRED_TOP_LEVEL_FILES:
        path = os.path.join(config_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Thieu file cau hinh bat buoc: {path}")
        paths[filename] = path
        
    rounds_dir = os.path.join(config_dir, _ROUNDS_SUBDIR)
    if not os.path.isdir(rounds_dir):
        raise FileNotFoundError(f"Thieu thu muc cau hinh bat buoc: {rounds_dir}")
    round_files = sorted(
        f for f in os.listdir(rounds_dir) if f.endswith(".yaml") or f.endswith(".yml")
    )
    if not round_files:
        raise FileNotFoundError(f"Thu muc {rounds_dir} khong chua file round nao (*.yaml)")

    base_data = _read_yaml(paths["base.yaml"])
    window_data = _read_yaml(paths["window.yaml"])
    training_data = _read_yaml(paths["training_defaults.yaml"])
    models_data = _read_yaml(paths["models.yaml"])

    base = _build_base_config(base_data, paths["base.yaml"])
    window = _build_window_config(window_data, paths["window.yaml"])
    training = _build_training_config(training_data, paths["training_defaults.yaml"])
    models = _build_models_config(models_data, paths["models.yaml"])
    
    rounds: Dict[str, RoundConfig] = {}
    for round_filename in round_files:
        round_path = os.path.join(rounds_dir, round_filename)
        round_data = _read_yaml(round_path)
        round_config = _build_round_config(round_data, round_path)
        rounds[round_config.round_id] = round_config

    return AppConfig(
        base=base,
        window=window,
        models=models,
        training=training,
        rounds=rounds
    )