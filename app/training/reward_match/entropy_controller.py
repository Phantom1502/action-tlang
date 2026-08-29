from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.config import EntropyConfig

DEFAULT_R_ENTROPY_FILENAME = "entropy_r_state.json" # r entropy, khuyến khích RR khác biệt
DEFAULT_ENTROPY_FILENAME = "entropy_state.json" # completions entropy, khuyến khích completions khác biệt
MIN_SAMPLES_PER_GROUP_FOR_ENTROPY = 2

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def shannon_entropy_nats(values: Sequence[str]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    return h


@dataclass
class EntropyState:
    ema_entropy: float
    bonus: float
    prev_error: float = 0.0


class EntropyController:
    """
    Thay EMABuffController cho task1 (chỉ còn 1 task zone). Khác EMA buff
    (đo tỉ lệ SUP/RES trên TOÀN BATCH so với target cố định), controller
    này đo ENTROPY của zone_type TRONG TỪNG ROLLOUT GROUP (các completion
    cùng 1 prompt) — bắt đúng hiện tượng model "chốt cứng" 1 hướng theo
    từng chart cụ thể (per-prompt collapse), điều mà EMA buff không thấy
    được vì nó chỉ nhìn tỉ lệ trung bình toàn batch (nhiều chart bù trừ
    lẫn nhau khiến tỉ lệ tổng thể trông vẫn cân bằng).

    Sàn 1 chiều (floor) — không có "tỉ lệ đúng" cố định như buff, chỉ cần
    đảm bảo advantage trong mỗi rollout group còn CÓ GÌ để so sánh.
    """

    def __init__(self) -> None:
        self.state: Optional[EntropyState] = None
        self._readings: List[float] = []

    def seed_from_round_config(self, entropy_config: EntropyConfig) -> None:
        self.state = EntropyState(
            ema_entropy=entropy_config.floor,
            bonus=0.0,
            prev_error=0.0,
        )
        self._readings.clear()

    def record_entropy(self, h: float) -> None:
        """Gọi 1 lần / rollout group (trong TLangReward.__call__) — tích
        luỹ, CHƯA update ngay. on_step_end() mới thật sự update PD."""
        self._readings.append(h)

    def on_step_end(self, entropy_config: EntropyConfig) -> None:        
        if self.state is None or not self._readings:
            self._readings.clear()
            return

        mean_h = sum(self._readings) / len(self._readings)
        st = self.state

        st.ema_entropy = (
            (1.0 - entropy_config.ema_alpha) * mean_h
            + entropy_config.ema_alpha * st.ema_entropy
        )

        error = max(0.0, entropy_config.floor - st.ema_entropy)
        d_error = error - st.prev_error
        st.prev_error = error

        if error > 0.0:
            delta = entropy_config.kp * error + entropy_config.kd * d_error
            delta = _clip(delta, -entropy_config.bonus_step_max, entropy_config.bonus_step_max)
            st.bonus = _clip(st.bonus + delta, 0.0, entropy_config.bonus_cap)
        else:
            # entropy đã ở/trên floor -> chủ động kéo bonus về 0 dần
            decay = min(st.bonus, entropy_config.bonus_step_max)
            st.bonus = max(0.0, st.bonus - decay)

        self._readings.clear()

    def get_bonus(self) -> float:
        return self.state.bonus if self.state is not None else 0.0

    def snapshot(self) -> Dict[str, float]:
        if self.state is None:
            return {}
        return {"ema_entropy": self.state.ema_entropy, "bonus": self.state.bonus, "prev_error": self.state.prev_error}

    def state_dict(self) -> Dict[str, float]:
        return self.snapshot()

    def load_state_dict(self, data: Dict[str, float]) -> None:
        self.state = EntropyState(
            ema_entropy=float(data["ema_entropy"]),
            bonus=float(data["bonus"]),
            prev_error=float(data.get("prev_error", 0.0)),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            self.load_state_dict(json.loads(p.read_text(encoding="utf-8")))
            return True
        except Exception:
            return False

    @classmethod
    def load_or_init(
        cls, 
        entropy_config: EntropyConfig, 
        file_name: str = DEFAULT_ENTROPY_FILENAME,
        resume_checkpoint: Optional[str] = None
    ) -> "EntropyController":
        controller = cls()
        state_path = os.path.join(resume_checkpoint, file_name) if resume_checkpoint else None
        loaded = bool(state_path and Path(state_path).exists() and controller.load(state_path))
        if not loaded:
            controller.seed_from_round_config(entropy_config)
        return controller