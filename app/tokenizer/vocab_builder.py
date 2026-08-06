from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

SPECIAL_TOKENS_IN_ID_ORDER: List[str] = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]  # id 0,1,2,3

ACTION_TYPES: Tuple[str, ...] = (
    "BUY", "SELL", "HOLD",
)

TREND_VALUES: Tuple[str, ...] = ("UP", "DOWN", "RANGE")

CANDLE_PREFIXES: Tuple[str, ...] = ("O", "H", "L", "C")  # <O_x> <H_x> <L_x> <C_x>


@dataclass(frozen=True)
class VocabGroup:
    """1 nhóm token trong bảng vocab — chỉ để in báo cáo `describe()`, không ảnh hưởng id."""
    name: str
    tokens: List[str]


def _build_groups(bin_min: int, bin_max: int, rr_min: int, rr_max: int) -> List[VocabGroup]:
    """
    Liệt kê tường minh mọi nhóm token, ĐÚNG THỨ TỰ sẽ được gán id (sau
    4 special token). Thứ tự này là 1 phần của "vocab contract" — đổi
    thứ tự đồng nghĩa đổi id, PHẢI rebuild lại mọi tokenizer.json/model
    embedding đã train nếu đổi. Xem thêm docs/tokenizer_v0.1.md mục 3.
    """
    groups: List[VocabGroup] = []

    # 1) Structural tags — chart/think/action wrapper
    groups.append(VocabGroup("structural_tags", [
        "<chart>", "</chart>",
        "<think>", "</think>",
        "<action>", "</action>",
    ]))

    # 2) Chart OHLC — ATOMIC, mỗi field 1 dải riêng [BIN_MIN, BIN_MAX].
    #    Model chỉ cần ĐỌC chart, không làm số học trực tiếp trên OHLC
    #    (spec mục 3) -> gộp tag+giá trị thành 1 token, không digit-decompose.
    for prefix in CANDLE_PREFIXES:
        groups.append(VocabGroup(
            f"candle_{prefix}",
            [f"<{prefix}_{v}>" for v in range(bin_min, bin_max + 1)],
        ))

    # 3) Trend — fused atomic (chỉ 3 giá trị, enum gộp hợp lý hơn tag+digit)
    groups.append(VocabGroup("trend", [f"<trend>{v}</trend>" for v in TREND_VALUES]))

    # 4) Tag mở/đóng cho field digit-decompose (current_price/zone) + SL label
    groups.append(VocabGroup("digit_field_tags", [
        "<current_price>", "</current_price>",
        "<zone_support>", "</zone_support>",
        "<zone_resistance>", "</zone_resistance>",
        "SL:",
    ]))

    # 5) Digit dùng CHUNG cho mọi field digit-decompose (current_price / zone
    #    lower / zone upper / SL) — cùng 1 không gian embedding 10 token, để
    #    model học so sánh/số học giữa các field này dễ hơn (spec mục 3).
    groups.append(VocabGroup("digit", [str(d) for d in range(10)]))

    # 6) Colon — phân cách 2 cạnh zone "lower:upper"
    groups.append(VocabGroup("colon", [":"]))

    # 8) Action type — atomic enum
    groups.append(VocabGroup("action_type", list(ACTION_TYPES)))

    # 9) RR — bracket-enum ATOMIC (range 1-9 quá nhỏ để digit-decompose)
    groups.append(VocabGroup("rr", [f"<RR_{v}>" for v in range(rr_min, rr_max + 1)]))

    return groups


def build_vocab(bin_min: int, bin_max: int, rr_min: int, rr_max: int) -> Dict[str, int]:
    """
    Trả về dict token_str -> id, id 0..3 = special token cố định, phần
    còn lại gán tuần tự theo thứ tự nhóm trong `_build_groups()`.

    Idempotent & deterministic: gọi lại nhiều lần luôn ra đúng 1 kết quả
    (không phụ thuộc hash-order hay random) — điều kiện bắt buộc để
    tokenizer.json build lại từ source vẫn tương thích checkpoint cũ.
    """
    vocab: Dict[str, int] = {}
    next_id = 0

    for tok in SPECIAL_TOKENS_IN_ID_ORDER:
        vocab[tok] = next_id
        next_id += 1

    for group in _build_groups(bin_min, bin_max, rr_min, rr_max):
        for tok in group.tokens:
            if tok in vocab:
                raise ValueError(f"Token trùng lặp giữa các nhóm: {tok!r} (nhóm {group.name})")
            vocab[tok] = next_id
            next_id += 1

    return vocab


def describe_vocab(bin_min: int, bin_max: int, rr_min: int, rr_max: int) -> str:
    """In bảng thống kê số token mỗi nhóm — dùng để đối chiếu với bảng
    vocab trong docs/train_pipeline_v0.1.md mục 2.2 (kỳ vọng ~4146 token)."""
    lines = [f"{'special':<20} count={len(SPECIAL_TOKENS_IN_ID_ORDER)}"]
    total = len(SPECIAL_TOKENS_IN_ID_ORDER)
    for group in _build_groups(bin_min, bin_max, rr_min, rr_max):
        lines.append(f"{group.name:<20} count={len(group.tokens)}")
        total += len(group.tokens)
    lines.append(f"{'TOTAL':<20} count={total}")
    return "\n".join(lines)


if __name__ == "__main__":
    from app.config.loader import load_config
    from app.config.schema import AppConfig

    cfg: AppConfig = load_config("configs")
    vocab = build_vocab(cfg.base.bin_min, cfg.base.bin_max, cfg.base.rr_min, cfg.base.rr_max)
    print(describe_vocab(cfg.base.bin_min, cfg.base.bin_max, cfg.base.rr_min, cfg.base.rr_max))
    print(f"\nlen(build_vocab()) = {len(vocab)}")
    assert len(vocab) == len(set(vocab.values())), "id bị trùng — bug trong build_vocab()"