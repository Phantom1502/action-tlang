import numpy as np
import os
from typing import List, Dict, Tuple, Any, Optional

from app.config import AppConfig, get_scale
from app.inference.model_inference import ModelInference
from tlang import (
    ChartCodec,
    ChartNode,  
    
    ASTVisitor,
)
from app.training.reward import TLangReward, StatsCollector
from app.training.reward.stats_collector import TaskRolloutMeta

class GRPOEval:
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
        self.stats = StatsCollector()
        self.reward = TLangReward(cfg, self.stats)
        
    def process_eval(
        self,
        batch: Dict[str, Any],
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
            
            prompts.append(prompt)
            future_bins.append(future_candles)
            symbols.append(symbol_tf)
         
        results = []
        completions = self.model.generate_batch([{"prompt": p} for p in prompts])
        for symbol_tf, prompt, completion, future_bin in zip(symbols, prompts, completions, future_bins):
            reward, meta = self.reward.compute_reward(prompt, completion, future_bin)
            self.stats.log(meta)
            results.append(reward)
            
        return results
        
    def eval(
        self, 
        dataset: Any,
        batch_size: int = 8
    ):
        rewards = []
        for batch in dataset.iter(batch_size=batch_size):
            rewards.extend(self.process_eval(batch))
        
        reward = np.mean(rewards)
        
        print(f"reward = {reward}")
        self.stats.print_summary()
        self.stats.save_summary_log("./out/grpo_eval.log")

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
    val_ds = val_ds.select(range(64))
    
    grpo_eval = GRPOEval(
        cfg, 
        "sullivan1502/base-action-pretrain",
        model_revision="afa956a788e6b01e2404ba7fed38e9963a96fdca", #r450
        model_subset="last-checkpoint",
    )
    grpo_eval.eval(val_ds, batch_size=8)