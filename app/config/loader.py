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
)

# Cac file bat buoc phai co truc tiep trong config_dir (khong ke rounds/, duoc xu ly rieng).
_REQUIRED_TOP_LEVEL_FILES = (
    "base.yaml",
    "window.yaml",
    "models.yaml",
    "training_defaults.yaml",
)

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
        zone_score_weight=_require_field(data, "zone_score_weight", source),
        zone_extend_multiplier=_require_field(data, "zone_extend_multiplier", source),
        zone_last_n_touch=_require_field(data, "zone_last_n_touch", source),
        sl_min_dist_bins=_require_field(data, "sl_min_dist_bins", source),
        sl_max_dist_bins=_require_field(data, "sl_max_dist_bins", source),
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

    base_data = _read_yaml(paths["base.yaml"])
    window_data = _read_yaml(paths["window.yaml"])
    training_data = _read_yaml(paths["training_defaults.yaml"])
    models_data = _read_yaml(paths["models.yaml"])

    base = _build_base_config(base_data, paths["base.yaml"])
    window = _build_window_config(window_data, paths["window.yaml"])
    training = _build_training_config(training_data, paths["training_defaults.yaml"])
    models = _build_models_config(models_data, paths["models.yaml"])

    return AppConfig(
        base=base,
        window=window,
        models=models,
        training=training
    )