import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq

from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from app.config import AppConfig, get_scale
from app.inference.model_inference import ModelInference
from tlang import (
    ChartCodec,
    ChartNode,  
    
    ASTVisitor,
)
from app.training.reward import TLangReward

class GRPODatasetBuilder:
    def __init__(
        self, 
        cfg: AppConfig,
        model_repo: str,
        model_subset: Optional[str] = None,
        model_revision: Optional[str] = None,
    ):
        self.cfg = cfg
        self.model: ModelInference = ModelInference(
            model_repo, 
            model_revision, 
            model_subset,
            max_new_tokens=36,
            do_sample=True,
        )
        self.visitor = ASTVisitor(digit_pad=cfg.tlang.digit_pad)
        self.reward = TLangReward(cfg)
        
    def process_gen_grpo(
        self,
        batch: Dict[str, Any],
        min_std: float = 0.5,
        n_samples: int = 16
    ):
        prompts = []
        completions = []
        future_bins = []
        symbols = []
        
        batch_size = len(batch["symbol"])
        for i in range(batch_size):
            symbol_tf = batch["symbol"][i]
            input_window = np.array(batch["input_window"][i], dtype=np.float32)
            future_window = np.array(batch["future_window"][i], dtype=np.float32)
            atr_100 = batch["atr_100"][i]
            symbol = symbol_tf.split("_")[0]
            timeframe = symbol_tf.split("_")[1]
            scale = get_scale(self.cfg, symbol, timeframe)
            
            codec = ChartCodec(scale, self.cfg.tlang.n_bins)
            input_candles, anchor_open = codec._encode_input(input_window, atr_100)
            future_candles = codec._encode_future(future_window, anchor_open, atr_100)
            
            prompt = self.visitor.render_chart_block(input_candles)
            
            prompts.extend([prompt] * n_samples)
            future_bins.extend([future_candles] * n_samples)
            symbols.extend([symbol_tf] * n_samples)
         
        prompt_groups = defaultdict(list)
        completions = self.model.generate_batch([{"prompt": p} for p in prompts])
        for symbol_tf, prompt, completion, future_bin in zip(symbols, prompts, completions, future_bins):
            reward, meta = self.reward.compute_reward(prompt, completion, future_bin)
            prompt_groups[prompt].append({
                "reward": reward,
                "symbol_tf": symbol_tf,
                "future_bin": future_bin,
            })
            
        # 4. Tính Mean, Std và lọc Prompt
        selected_grpo_samples = []
        for prompt, samples in prompt_groups.items():
            rewards = [s["reward"] for s in samples]
            
            # Tính mean và std
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))

            # --- ĐIỀU KIỆN LỌC GRPO ---
            # Chỉ lấy các prompt có std cao hơn ngưỡng (Model có sự phân hóa tốt)
            if std_reward >= min_std:
                selected_grpo_samples.append({
                    "symbol": samples[0]["symbol_tf"],
                    "prompt": prompt,
                    "future_bins": [[int(candle.open), int(candle.high), int(candle.low), int(candle.close)] for candle in samples[0]["future_bin"]],
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                })
            
        return selected_grpo_samples
        
    def save_grpo_parquet(self, records: list, output_path: str):
        if not records:
            print("⚠️ Không có record nào để lưu!")
            return

        # 1. Định nghĩa Schema cố định, định kiểu rõ ràng từng trường
        schema = pa.schema([
            ("symbol", pa.string()),
            ("prompt", pa.string()),
            ("future_bins", pa.list_(pa.list_(pa.int32()))),
            ("mean_reward", pa.float32()),
            ("std_reward", pa.float32())
        ])

        # 3. Chuyển thành PyArrow Table theo đúng Schema đã định nghĩa
        table = pa.Table.from_pylist(records, schema=schema)

        # 4. Ghi ra file Parquet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")
        
        print(f"✅ Đã lưu Dataset chuẩn Schema vào: {output_path}")
        
    def gen(
        self, 
        dataset: Any,
        limit: int = 1000,
        batch_size: int = 8,
        min_std: float = 0.5,
        n_samples: int = 16,
        output_path: str = "./out/grpo_dataset.parquet"
    ):
        records = []
        for batch in dataset.iter(batch_size=batch_size):
            records.extend(self.process_gen_grpo(batch, min_std=min_std, n_samples=n_samples))
            
            print(f"✅ Đã tạo {len(records)} mẫu grpo")
            
            if len(records) >= limit:
                break
            
        if not records:
            print("⚠️ Không có mẫu nào đạt điều kiện!")
            return
        
        self.save_grpo_parquet(records, output_path)

if __name__ == "__main__":
    from datasets import load_dataset
    from app.config import AppConfig, load_config
    cfg: AppConfig = load_config("configs")
    
    data_files = {
        "train": "data/slide_window/grpo/window_200_train.parquet",
        "val": "data/slide_window/grpo/window_200_train.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    val_ds = dataset["val"]
    val_ds = val_ds.select(range(4))
    
    grpo_eval = GRPODatasetBuilder(
        cfg, 
        "sullivan1502/base-action-pretrain",
        model_revision="afa956a788e6b01e2404ba7fed38e9963a96fdca", #r450
        model_subset="last-checkpoint",
    )
    grpo_eval.gen(
        val_ds, 
        limit=1280, 
        batch_size=8,
        min_std=0.5,
        n_samples=16,
        output_path="./out/grpo_dataset.parquet",
    )
    
    # review dataset
    dataset = load_dataset("parquet", data_files={"train": "./out/grpo_dataset.parquet"})
    print(dataset['train'][0])