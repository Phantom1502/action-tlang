from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset

from app.config import AppConfig
from app.training.reward import (
    StatsCollector,
    TLangReward,
)
from app.inference import ModelInference
from app.lang import CandleNode

ZONE_TYPES = ("support", "resistance")
RR_TYPES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
REWARD_RELEVANT_ZONE_TYPES = ("SUP_ZONE", "RES_ZONE")   # NO_ZONE trung tính, bỏ qua ở mean_reward/touch_rate


def print_zone_quality_histogram(
    stats_collector: StatsCollector,
    zone_score_weight: float,
    rr_max: int,
    zone_types: Sequence[str] = REWARD_RELEVANT_ZONE_TYPES,
) -> None:
    """
    Bucket zone_quality (đã nhân weight, đúng thang với mean_reward) VỀ LẠI
    số R nguyên gần nhất (0R, 1R, 2R, ..., rr_maxR) bằng cách chia ngược
    cho zone_score_weight rồi round — group theo R để đọc trực quan.

    QUAN TRỌNG: TÁCH RIÊNG "NOT_TOUCHED" (zone không bao giờ được giá
    tương lai chạm tới trong outcome_horizon) khỏi bucket "0R" (zone CÓ
    chạm nhưng thua ngay khi vừa vào lệnh) — 2 cái này CÙNG có
    zone_quality=0.0 nên trước khi có field is_touched sẽ bị gộp lẫn vào
    nhau, đọc sai hoàn toàn ý nghĩa (0R cao có thể là "model hay đặt zone
    không ai chạm tới" chứ không phải "model hay chọn sai hướng rồi bị
    quét SL" — 2 vấn đề cần 2 hướng sửa khác hẳn nhau).
    """
    if zone_score_weight <= 0:
        print("zone_score_weight <= 0 — không thể quy đổi ngược về R, bỏ qua histogram.")
        return

    per_type_r_counts = {zt: defaultdict(int) for zt in zone_types}
    per_type_not_touched = {zt: 0 for zt in zone_types}
    per_type_total = {zt: 0 for zt in zone_types}

    for r in stats_collector._records:
        if r.zone_type not in zone_types or r.zone_quality is None:
            continue
        per_type_total[r.zone_type] += 1
        if r.is_touched is False:
            per_type_not_touched[r.zone_type] += 1
            continue
        r_multiple_approx = round(r.zone_quality / zone_score_weight)
        r_multiple_approx = max(0, min(rr_max, r_multiple_approx))   # kẹp về [0, rr_max] phòng sai số round
        per_type_r_counts[r.zone_type][r_multiple_approx] += 1

    print("\n=== Phân phối zone_quality theo bội số R (SUP/RES) ===")
    for zt in zone_types:
        total = per_type_total[zt]
        print(f"\n{zt} (n={total}):")
        if total == 0:
            print("  (không có sample nào)")
            continue

        n_not_touched = per_type_not_touched[zt]
        ratio_not_touched = n_not_touched / total
        bar_nt = "#" * int(ratio_not_touched * 40)
        print(f"  KHÔNG CHẠM  count={n_not_touched:<6} ratio={ratio_not_touched * 100:5.1f}%  {bar_nt}")

        for r_level in range(0, rr_max + 1):
            n = per_type_r_counts[zt].get(r_level, 0)
            ratio = n / total
            bar = "#" * int(ratio * 40)
            print(f"  {r_level:>2}R  count={n:<6} ratio={ratio * 100:5.1f}%  {bar}")

        n_at_cap = per_type_r_counts[zt].get(rr_max, 0)
        n_zero_touched = per_type_r_counts[zt].get(0, 0)
        print(f"  -> tỉ lệ chạm cap ({rr_max}R): {n_at_cap / total * 100:.1f}%")
        print(f"  -> tỉ lệ CHẠM zone nhưng thua ngay (0R, KHÁC 'không chạm'): {n_zero_touched / total * 100:.1f}%")


class ActionEval:
    def __init__(
        self,
        cfg: AppConfig,
        model_repo: str,
        dataset_repo: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        split: str = "val",
        tokenizer_repo: Optional[str] = None,
        batch_size: int = 16,
        max_new_tokens: int = 10,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
        limit: Optional[int] = None,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        
        self.model: ModelInference = ModelInference(
            model_repo, 
            revision, 
            subfolder, 
            tokenizer_repo, 
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_p
        )

        self.dataset = load_dataset(dataset_repo, split=split)
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        # --- Eval mode: buff_controller=None -> TLangReward tự bỏ hẳn phần
        # buff (reward = gate_score + zone_quality), KHÔNG cần dựng
        # EMABuffController giả lập = 0 nữa (TLangReward đã tự xử lý case
        # này qua tham số buff_controller Optional). round_config KHÔNG
        # cần truyền nữa — zone_score_weight đã chuyển hẳn vào
        # cfg.base.zone_score_weight (xem tlang_reward.py). ---
        self.stats_collector = StatsCollector()
        self.reward_fn = TLangReward(cfg, stats_collector=self.stats_collector)

        # Lưu song song reward TRẢ VỀ THẬT của compute_reward() theo đúng thứ tự
        # log — KHÔNG suy ngược từ hằng số gate_score=2.0 (giả định nội bộ của
        # TLangReward có thể đổi sau này, tránh phụ thuộc ngầm vào con số đó).
        self._rewards: List[float] = []

    def run(self) -> Dict[str, Any]:
        n = len(self.dataset)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            rows = [self.dataset[i] for i in range(start, end)]
            completions = self.model.generate_batch(rows)

            for row, completion in zip(rows, completions):
                future_bins_text = row["future_bins"]
                future_bins: List[CandleNode] = [CandleNode(open=c[0], high=c[1], low=c[2], close=c[3]) for c in future_bins_text]
                reward, meta = self.reward_fn.compute_reward(row["prompt"], completion, future_bins)
                self.stats_collector.log(meta)
                self._rewards.append(reward)

            print(f"  ... {end}/{n}")

        return self.summarize()

    # ------------------------------------------------------------------
    # Thống kê — 3 thành phần theo đúng yêu cầu.
    # ------------------------------------------------------------------
    def summarize(self):
        records = self.stats_collector._records
        n = len(records)
        n_wf = sum(1 for r in records if r.well_formed)
        n_sem = sum(1 for r in records if r.well_formed and r.semantic_passed)
        
        print(f"=== ActionEval summary ===")
        print(f"n_samples = {n}")
        print(f"well_form_rate = {n_wf / n * 100:.1f}%")
        print(f"semantic_pass_rate (trong số well-formed) = {n_sem / n_wf * 100:.1f}%")
        
        by_zone_total: Dict[str, int] = defaultdict(int)
        by_rr_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        raw: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {
                        "count": 0, "entry_qualities": [], "outcomes": []
                    }
                )
            )
        )
        
        for r in records:
            if not (r.well_formed and r.semantic_passed) or r.zone_type is None:
                continue
            by_zone_total[r.zone_type] += 1
            by_rr_total[r.zone_type][r.rr] += 1
            entry = raw[r.zone_type][r.action_type][r.rr]
            entry["count"] += 1
            if r.entry_quality is not None:
                entry["entry_qualities"].append(r.entry_quality)
            if r.outcome is not None:
                entry["outcomes"].append(r.outcome)

        zone_type_ratio = {zt: by_zone_total[zt] / n for zt in ZONE_TYPES}
        print("\n-- Tī lệ zone_type (trong số được pass gate) --")
        for zt in ZONE_TYPES:
            ratio = zone_type_ratio[zt]
            print(f"  {zt:<10} ratio={ratio * 100:5.1f}%")
            
        rr_type_ratio = {rr: {zt: by_rr_total[zt][rr] / by_zone_total[zt] for zt in ZONE_TYPES} for rr in RR_TYPES}
        print("\n-- Tī lệ rr_type (trong từng loại zone type đã pass gate)--")
        for zt in ZONE_TYPES:
            print(f"  {zt:<10}")
            for rr in RR_TYPES:
                action_type = "BUY" if zt == "support" else "SELL"
                ratio = rr_type_ratio[rr][zt]
                count = raw[zt][action_type][rr]["count"]
                avg_quality = sum(raw[zt][action_type][rr]["entry_qualities"]) / count if count > 0 else 0
                avg_outcome = sum(raw[zt][action_type][rr]["outcomes"]) / count if count > 0 else 0
                print(f"    RR_{rr:<10} ratio={ratio * 100:5.1f}% count={count:5d} avg_quality={avg_quality:5.2f} avg_outcome={avg_outcome:5.2f}")
                
if __name__ == "__main__":
    from app.config import AppConfig, load_config
    cfg: AppConfig = load_config("configs")
    action_eval = ActionEval(
        cfg,
        model_repo="sullivan1502/base-action-grpo",
        dataset_repo="sullivan1502/action-data",
        limit=5000
    )
    action_eval.run()