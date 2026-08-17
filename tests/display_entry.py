from datasets import load_dataset
from typing import List, Optional
from app.inference import ModelInference
from app.config import AppConfig, load_config
from app.training.reward import TLangReward, derive_target

from app.lang import (
    Parser,
    ParseResult,
    ProgramNode,
    CandleNode
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot(program: ProgramNode, future_bins: List[CandleNode], zone_score: float):
    all_candles: List[CandleNode] = program.chart.candles + future_bins
    n_input = len(program.chart.candles)

    fig, ax = plt.subplots(figsize=(16, 6))
    
    for i, cnd in enumerate(all_candles):
        color = "tab:green" if cnd.close >= cnd.open else "tab:red"
        ax.plot([i, i], [cnd.low, cnd.high], color=color, linewidth=1)
        lower_body = min(cnd.open, cnd.close)
        height = max(abs(cnd.close - cnd.open), 1)  # tối thiểu 1 bin cho dễ nhìn nếu doji
        ax.add_patch(patches.Rectangle((i - 0.3, lower_body), 0.6, height, color=color))
        
    think = program.think
    # --- Zone (support/resistance) ---
    if think.zone is not None:
        zone_color = "tab:blue" if think.zone.direction == "support" else "tab:orange"
        ax.axhspan(
            think.zone.lower_bin, think.zone.upper_bin, color=zone_color, alpha=0.15,
            label=f"zone_{think.zone.direction} [{think.zone.lower_bin}:{think.zone.upper_bin}]",
        )

    # --- current_price ---
    ax.axhline(
        think.current_price_bin, color="black", linestyle="--", linewidth=1,
        label=f"current_price={think.current_price_bin}",
    )

    # --- Đường dọc đánh dấu nến hiện tại (ranh giới input/future) ---
    current_idx = n_input - 1
    ax.axvline(
        current_idx, color="black", linestyle="-", linewidth=1.2, alpha=0.7,
        label=f"nến hiện tại (idx={current_idx})",
    )
    
    action = program.action
    if action.action_type in ("BUY", "SELL") and action.sl is not None and action.rr is not None:
        entry = think.current_price_bin
        direction = "long" if action.action_type == "BUY" else "short"
        target = derive_target(entry, action.sl, action.rr, direction)

        ax.axhline(action.sl, color="red", linestyle=":", linewidth=1.5, label=f"SL={action.sl}")
        if target is not None:
            ax.axhline(
                target, color="tab:green", linestyle=":", linewidth=1.5,
                label=f"TP(RR{action.rr})={target}",
            )

    ax.set_title(
        f"trend={think.trend} action={action.action_type} zone_score={zone_score:.2f} "
        f"sl={action.sl} rr={action.rr}"
    )
    ax.set_xlabel("candle index")
    ax.set_ylabel("bin")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    plt.show()

def parse(cfg: AppConfig, text: str) -> Optional[ProgramNode]:
    parser_result: ParseResult = Parser.from_text(cfg, text).parse()
    return parser_result.ast
    
if __name__ == "__main__":
    cfg: AppConfig = load_config("configs")
    
    ds_idx = 3
    # Lấy mẫu
    dataset = load_dataset("parquet", data_files={"train": "data/dataset/val_llm.parquet"})
    prompt = dataset['train'][ds_idx]['prompt']
    future_bins_text = dataset['train'][ds_idx]['future_bins']
    zone_score = dataset['train'][ds_idx]['zone_score']
    future_bins: List[CandleNode] = [CandleNode(open=c[0], high=c[1], low=c[2], close=c[3]) for c in future_bins_text]
    
    tlang: TLangReward = TLangReward(cfg)
    
    # Inference
    inference = ModelInference(model_repo="sullivan1502/base-action-grpo")
    completions: List[str] = inference.generate_one(prompt, n_gen=16)
    
    from collections import Counter
    
    counter = Counter()
    for completion in completions:
        reward, meta = tlang.compute_reward(prompt, completion, future_bins)
        
        counter[meta.action_type] += 1
        
        if meta.well_formed and meta.semantic_passed and meta.action_type != "HOLD":
            #print(f"Reward: {reward}, Meta: {meta}")
            program = parse(cfg, prompt + " " + completion)
            plot(program, future_bins, zone_score)
            
    print(counter)
    