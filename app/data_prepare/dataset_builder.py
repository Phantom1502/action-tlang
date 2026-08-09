from __future__ import annotations

import random
from typing import List, Optional, Tuple

from app.config import AppConfig, load_config
from app.data_prepare.generator import ActionGenerator
from app.lang import (
    ProgramNode,
    Parser,
    ParseResult,
    CandleNode
)
from .helper import DataAugmenter

class DatasetBuilder:
    def __init__(self, cfg: AppConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        
        self.generator = ActionGenerator(cfg, seed=seed)
        self.augmenter = DataAugmenter(self.rng, self.cfg.base.n_bins)
        
    def build_rows(
        self,
        program: ProgramNode,
        symbol: str,
        window_id: str,
        zone_score: float,
        n_augments: int = 1,
    ):
        programs: List[Tuple[str, ProgramNode]] = [(window_id, program)]
        if n_augments > 0:
            variants = self.augmenter.augment_shift(program, n_augments=n_augments)
            for aug_shift, variant in variants:
                programs.append((f"{window_id}_{aug_shift}", variant))
        
        samples = []
        for window_id, program in programs:
            row = self.generator.generate_one(program)
            if row is None:
                continue
            samples.append((window_id, row, program.future_bins))
        return [
            {
                "prompt": s.prompt, 
                "completion": s.completion,
                "future_bins": [[c.open, c.high, c.low, c.close] for c in future_bins],
                "symbol": symbol,
                "zone_score": zone_score,
                "window_id": window_id,
            } 
            for window_id, s, future_bins in samples
        ]
    
def main(
    cfg: AppConfig, 
    input_dir: str,
    output_dir: str,
    seed: Optional[int] = None,
    n_augments = 10,
):
    from datasets import load_dataset
    import hashlib

    data_files = {
        "train": f"{input_dir}/train.parquet",
        "val": f"{input_dir}/val.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    
    def preprocess_for_llm(batch):
        prompts = []
        completions = []
        future_bins_list = []
        symbols = []
        zone_scores = []
        window_ids = []
        
        batch_size = len(batch["prompt"])
        
        # Duyệt qua các phần tử trong batch (chạy trong RAM của batch đó, cực nhẹ)
        for i in range(batch_size):
            prompt = batch["prompt"][i]
            completion = batch["completion"][i]
            future_bins = batch["future_bins"][i]
            symbol = batch["symbol"][i]
            window_id = batch["window_id"][i]
            zone_quality = batch["zone_quality"][i]
            
            row_seed = int(hashlib.md5(window_id.encode()).hexdigest(), 16) % (2**31)
            builder = DatasetBuilder(cfg, seed=row_seed)
            
            parse_result: ParseResult = Parser.from_text(cfg, prompt + " " + completion).parse()
            parse_result.ast.future_bins = [CandleNode(open=c[0], high=c[1], low=c[2], close=c[3]) for c in future_bins]
            records = builder.build_rows(
                parse_result.ast,
                symbol,
                window_id,
                zone_quality,
                n_augments=n_augments
            )
                        
            for record in records:
                prompts.append(record["prompt"])
                completions.append(record["completion"])
                future_bins_list.append(record["future_bins"])
                symbols.append(record["symbol"])
                zone_scores.append(record["zone_score"])
                window_ids.append(record["window_id"])
        # Trả về các cột mới cho Dataset LLM
        return {
            "prompt": prompts,
            "completion": completions,
            "future_bins": future_bins_list,
            "symbol": symbols,
            "zone_score": zone_scores,
            "window_id": window_ids
        }
        
    llm_dataset = dataset.map(
        preprocess_for_llm,
        batched=True,
        batch_size=2000, # Mỗi lần nạp 2000 dòng vào RAM để parse
        num_proc=4,      # Số lượng nhân CPU chạy song song
        remove_columns=dataset["train"].column_names # Xóa các cột gốc (id, type, score...) để thu gọn dataset
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    llm_dataset["train"].shuffle(seed=seed).to_parquet(f"{output_dir}/train_llm.parquet")
    llm_dataset["val"].to_parquet(f"{output_dir}/val_llm.parquet")    

if __name__ == '__main__':
    cfg: AppConfig = load_config("configs")
    main(cfg, "data/filter", "data/dataset", seed=42, n_augments=4)