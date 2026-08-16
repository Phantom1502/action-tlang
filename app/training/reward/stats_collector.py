from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def stats_path_for_rank(output_dir: str, round_id: str, rank: int) -> str:
    """NGUỒN DUY NHẤT cho quy ước đặt tên file stats — dùng chung bởi cả
    save-side (StatsPersistCallback.on_save/on_train_end) LẪN load-side
    (train_grpo.py lúc resume). KHÔNG định nghĩa lại công thức này ở nơi
    khác — đổi 1 chỗ, mọi nơi ăn theo."""
    return os.path.join(output_dir, f"{round_id}_stats_rank{rank}.json")


@dataclass
class TaskRolloutMeta:
    well_formed: bool
    semantic_passed: bool
    zone_type: Optional[str]          # "support" / "resistance" — None nếu chưa pass gate
    action_type: Optional[str]        # "BUY" / "SELL" / "HOLD" — None nếu chưa pass gate
    entry_quality: Optional[float]    # None nếu chưa pass gate
    outcome: Optional[float]          # None nếu chưa pass gate
    sl: Optional[int] = None          # CHỈ có ở BUY/SELL — None nếu chưa pass gate, 0 neu HOLD
    rr: Optional[int] = None          # CHỈ có ở BUY/SELL — None nếu chưa pass gate, 0 neu HOLD
    entropy: Optional[float] = None


class StatsCollector:
    """
    Nguồn DUY NHẤT cho cả report (print_summary(), gọi theo nhịp save_steps)
    LẪN nuôi buff (counts_since_step_boundary(), gọi theo nhịp optimizer
    step). Buff task2 tách theo TỪNG zone_type riêng (EMABuffController
    task2 gọi on_step_end() 2 lần/step — 1 lần/zone_type) — vì vậy mọi hàm
    đếm ở đây đều nhận `zone_type` làm tham số bắt buộc, KHÔNG gộp chung
    như task1 (task1 chỉ 1 chiều zone_type, task2 2 chiều zone_type×action_type).

    mark_step_boundary() chỉ dịch 1 con trỏ index, KHÔNG xoá gì — reset()
    (gọi ở on_save, cùng nhịp save_steps) mới thật sự xoá records VÀ đưa
    watermark về 0.
    """

    def __init__(self) -> None:
        self._records: List[TaskRolloutMeta] = []
        self._step_boundary: int = 0

    def log(self, meta: TaskRolloutMeta) -> None:
        self._records.append(meta)

    def reset(self) -> None:
        self._records.clear()
        self._step_boundary = 0

    def mark_step_boundary(self) -> None:
        self._step_boundary = len(self._records)

    @staticmethod
    def _filter_and_count(records: Sequence[TaskRolloutMeta], zone_type: str, key_fn) -> Tuple[Dict[str, int], int]:
        """CHỈ đếm record đã pass gate (well_formed + semantic_passed) VÀ
        đúng zone_type — khớp quy ước "buff chỉ tính sau khi pass gate",
        và mỗi zone_type có buff riêng nên phải lọc trước khi đếm."""
        counts: Dict[str, int] = defaultdict(int)
        total = 0
        for r in records:
            if not r.well_formed or not r.semantic_passed:
                continue
            if r.zone_type != zone_type:
                continue
            key = key_fn(r)
            if key is None:
                continue
            counts[key] += 1
            total += 1
        return dict(counts), total

    def counts_since_step_boundary(self, zone_type: str, key_fn) -> Tuple[Dict[str, int], int]:
        print(f"[debug] zone_type values seen this window: {Counter(r.zone_type for r in self._records[self._step_boundary:])}")
        """Dùng để nuôi buff — CHỈ đếm records kể từ watermark step trước."""
        return self._filter_and_count(self._records[self._step_boundary:], zone_type, key_fn)

    def full_history_counts(self, zone_type: str, key_fn) -> Tuple[Dict[str, int], int]:
        """Dùng cho report — đếm TOÀN BỘ records kể từ lần reset() gần nhất."""
        return self._filter_and_count(self._records, zone_type, key_fn)

    def summary(self) -> Dict[str, Dict[str, dict]]:
        """
        Breakdown theo zone_type -> action_type, CHỈ tính trên record đã
        pass gate (well_formed + semantic_passed). Bao gồm avg_rr +
        rr_distribution CHO BUY/SELL (HOLD không có RR, list rỗng -> None).
        """
        by_zone_total: Dict[str, int] = defaultdict(int)
        raw: Dict[str, Dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {
                "count": 0, "entry_qualities": [], "outcomes": [], "rrs": [], "entropy": []
            })
        )
        for r in self._records:
            if not (r.well_formed and r.semantic_passed) or r.zone_type is None:
                continue
            by_zone_total[r.zone_type] += 1
            entry = raw[r.zone_type][r.action_type]
            entry["count"] += 1
            if r.entry_quality is not None:
                entry["entry_qualities"].append(r.entry_quality)
            if r.outcome is not None:
                entry["outcomes"].append(r.outcome)
            if r.rr is not None:
                entry["rrs"].append(r.rr)
            if r.entropy is not None:
                entry["entropy"].append(r.entropy)

        result: Dict[str, Dict[str, dict]] = {}
        for zone_type, action_types in raw.items():
            result[zone_type] = {}
            total = by_zone_total[zone_type]
            for action_type, entry in action_types.items():
                eql = entry["entry_qualities"]
                outcomes = entry["outcomes"]
                rrs = entry["rrs"]
                entropies = entry["entropy"]
                result[zone_type][action_type] = {
                    "count": entry["count"],
                    "freq_within_zone": entry["count"] / total if total else 0.0,
                    "avg_entry_quality": (sum(eql) / len(eql)) if eql else None,
                    "avg_outcome": (sum(outcomes) / len(outcomes)) if outcomes else None,
                    "avg_entropy": (sum(entropies) / len(entropies)) if entropies else None,
                    "avg_rr": (sum(rrs) / len(rrs)) if rrs else None,
                    "rr_distribution": dict(sorted(Counter(rrs).items())) if rrs else None,
                }
        return result

    def print_summary(self) -> None:
        n = len(self._records)
        n_wf = sum(1 for r in self._records if r.well_formed)
        n_sem = sum(1 for r in self._records if r.well_formed and r.semantic_passed)

        print("=== StatsCollector summary (task2 — action) ===")
        print(f"n_records = {n}")
        if n:
            print(f"well_form_rate = {n_wf / n * 100:.1f}%")
        if n_wf:
            print(f"semantic_pass_rate (trong số well-formed) = {n_sem / n_wf * 100:.1f}%")

        print("\n-- Chi tiết theo zone -> action (đã pass gate) --")
        detail = self.summary()
        if not detail:
            print("  (chưa có record nào pass gate)")
        for zone_type, action_types in detail.items():
            print(f"zone_type={zone_type}")
            for action_type, stat in action_types.items():
                avg_eq = f"{stat['avg_entry_quality']:.3f}" if stat["avg_entry_quality"] is not None else "-"
                avg_out = f"{stat['avg_outcome']:.3f}" if stat["avg_outcome"] is not None else "-"
                avg_entropy = f"{stat['avg_entropy']:.3f}" if stat["avg_entropy"] is not None else "-"
                avg_rr = f"{stat['avg_rr']:.2f}" if stat.get("avg_rr") is not None else "-"
                line = (
                    f"  {action_type:<10} count={stat['count']}({stat['freq_within_zone']*100:5.1f}%)  "
                    f"ENTRY_QUALITY={avg_eq:>7} OUTCOME={avg_out:>7} avg_entropy={avg_entropy:>5} avg_RR={avg_rr:>5}"
                )
                dist = stat.get("rr_distribution")
                if dist:
                    dist_str = " ".join(f"{k}:{v}" for k, v in dist.items())
                    line += f"  rr_dist=[{dist_str}]"
                print(line)

    def to_list(self) -> List[dict]:
        return [asdict(r) for r in self._records]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"records": self.to_list()}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StatsCollector":
        collector = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("records", []):
                d.setdefault("rr", None)   # tương thích ngược file stats cũ chưa có field này
                d.setdefault("entropy", None)
                collector.log(TaskRolloutMeta(**d))
        return collector

    @classmethod
    def merge_from_files(cls, paths) -> "StatsCollector":
        collector = cls()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("records", []):
                d.setdefault("rr", None)
                d.setdefault("entropy", None)
                collector.log(TaskRolloutMeta(**d))
        return collector