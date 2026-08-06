from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from app.config import (
    AppConfig,
    load_config,
)

from app.tokenizer.vocab_builder import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocab,
)


def build_raw_tokenizer(
    bin_min: int, 
    bin_max: int, 
    rr_min: int, 
    rr_max: int
) -> Tokenizer:
    vocab = build_vocab(bin_min, bin_max, rr_min, rr_max)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    bos_id = vocab[BOS_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    # Thêm <bos>/<eos> tự động quanh completion khi encode 1 chuỗi đơn —
    # khớp LlamaConfig(bos_token_id=1, eos_token_id=2) trong
    # docs/train_pipeline_v0.1.md mục 1.1. Không thêm token đặc biệt nào
    # khác (không có [CLS]/[SEP] kiểu BERT — kiến trúc là causal LM).
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    return tokenizer


def build_fast_tokenizer(
    bin_min: int,
    bin_max: int,
    rr_min: int,
    rr_max: int,    
) -> PreTrainedTokenizerFast:
    raw_tokenizer = build_raw_tokenizer(bin_min, bin_max, rr_min, rr_max)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
    )
    return fast


def main(cfg: AppConfig) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir", type=str, default="./tokenizer_out",
        help="Thư mục lưu tokenizer (tokenizer.json + tokenizer_config.json ...)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = build_fast_tokenizer(
        bin_min=cfg.base.bin_min,
        bin_max=cfg.base.bin_max,
        rr_min=cfg.base.rr_min,
        rr_max=cfg.base.rr_max,
    )
    fast.save_pretrained(str(out_dir))

    print(f"Đã lưu tokenizer vào: {out_dir.resolve()}")
    print(f"vocab_size = {fast.vocab_size}")
    print(f"pad_token_id={fast.pad_token_id} bos_token_id={fast.bos_token_id} "
          f"eos_token_id={fast.eos_token_id} unk_token_id={fast.unk_token_id}")


if __name__ == "__main__":
    cfg: AppConfig = load_config("configs")
    main(cfg)