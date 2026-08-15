from __future__ import annotations
 
import os
import random
from typing import Any, Dict, List, Optional
 
from datasets import Dataset
 
from app.config.schema import AppConfig
from app.lang.ast_nodes import CandleNode, ChartNode, ThinkNode, ZoneNode
from app.lang.ast_visitor import ASTVisitor
from app.lang.parser import Parser
from app.data_prepare.generator import ActionGenerator
from app.data_prepare.helper import DataAugmenter

def _make_rng(window_id: str) -> random.Random:
    nonce = int.from_bytes(os.urandom(4), "big")
    seed = (hash(window_id) ^ nonce) & 0xFFFFFFFF
    return random.Random(seed)

def _gen_seed(window_id: str) -> int:
    nonce = int.from_bytes(os.urandom(4), "big")
    return (hash(window_id) ^ nonce) & 0xFFFFFFFF

class ActionSFTDataset:
    def __init__(
        self,
        base_dataset: Dataset,
        cfg: AppConfig,
        augment_prob: float = 0.5,
    ):
        if not (0.0 <= augment_prob <= 1.0):
            raise ValueError(f"augment_prob phải trong [0,1], nhận {augment_prob}")
        
        required_cols = {"prompt", "completion", "future_bins", "window_id"}
        missing = required_cols - set(base_dataset.column_names)
        if missing:
            raise ValueError(
                f"base_dataset thiếu cột bắt buộc: {missing} — cần đủ "
                f"{required_cols} (window_id dùng để seed RNG augment, "
                f"KHÔNG xuất hiện lại trong output)."
            )
 
        self.cfg = cfg
        self.augment_prob = augment_prob
        self._visitor = ASTVisitor(digit_pad=cfg.base.digit_pad)
        self.dataset = base_dataset.with_transform(self._transform_batch)
        
    def _transform_one(self, prompt: str, completion: str, future_bins: str, window_id: str) -> Dict[str, str]:
        seed_augment = _gen_seed(window_id)
        seed_generate = _gen_seed(window_id)
        generator = ActionGenerator(self.cfg, seed=seed_generate)
        parse_result = Parser.from_text(self.cfg, prompt).parse()
            
        program = parse_result.ast
        program.future_bins =  [CandleNode(open=c[0], high=c[1], low=c[2], close=c[3]) for c in future_bins]
        
        rng = random.Random(seed_augment)
        if rng.random() <= self.augment_prob:
            augmenter = DataAugmenter(rng, self.cfg.base.n_bins)
            aug_list = augmenter.augment_shift(program, n_augments=1)
            if aug_list and len(aug_list) > 0:
                _, program = aug_list[0]
        
        record = generator.generate_one(program)
        if record is None:
            print(f"Fallback: {prompt} {completion}")
            return {
                "prompt": prompt,
                "completion": completion,
            }
 
        print(f"Original: {completion} | Generated: {record.completion}")
        return {
            "prompt": record.prompt,
            "completion": record.completion,
        }
 
    def _transform_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        prompts: List[str] = []
        completions: List[str] = []
        for prompt, completion, future_bins, window_id in zip(batch["prompt"], batch["completion"], batch["future_bins"], batch["window_id"]):
            out = self._transform_one(prompt, completion, future_bins, window_id)
            prompts.append(out["prompt"])
            completions.append(out["completion"])
        return {"prompt": prompts, "completion": completions}
 
    def __len__(self) -> int:
        return len(self.dataset)
 
    def __getitem__(self, idx):
        return self.dataset[idx]
    
if __name__ == "__main__":
    from datasets import load_dataset
    from app.config import load_config, AppConfig
    cfg: AppConfig = load_config("configs")
    
    # Lấy duy nhất phần tử đầu tiên (index 0) dưới dạng Dataset
    base_dataset = load_dataset("sullivan1502/action-data", split="train").select(range(1))
    
    dataset: Dataset = ActionSFTDataset(base_dataset, cfg, augment_prob=0.0)
    for _ in range(10):
        next(iter(dataset))