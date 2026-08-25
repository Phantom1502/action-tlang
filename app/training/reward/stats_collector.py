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
    trend_type: Optional[str]         # "UP" / "DOWN" / "RANGE" — None nếu chưa pass gate
    zone_type: Optional[str]          # "support" / "resistance" — None nếu chưa pass gate
    action_type: Optional[str]        # "BUY" / "SELL" / "HOLD" — None nếu chưa pass gate
    outcome: Optional[float]          # None nếu chưa pass gate
    outcome_status: Optional[str]     # "WIN" / "LOSS" / "WIN_1R" / "ENTRY_TIMEOUT" / "TIMEOUT" / "INVALID_SETUP"
    rr: Optional[int] = None          # CHỈ có ở BUY/SELL — None nếu chưa pass gate, 0 neu HOLD


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
        by_trend_total: Dict[str, int] = defaultdict(int)
        by_zone_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        raw: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict( # trend_type -> zone_type -> action_rr (BUY_1 : BUY WITH RR 1)
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {
                        "count": 0, "outcomes": [], "status": [],
                    }
                )
            )
        )
        for r in self._records:
            if not (r.well_formed and r.semantic_passed) or r.trend_type is None:
                continue
            by_trend_total[r.trend_type] += 1
            by_zone_total[r.trend_type][r.zone_type] += 1
            entry = raw[r.trend_type][r.zone_type][f"{r.action_type}_{r.rr}"]
            entry["count"] += 1
            if r.outcome is not None:
                entry["outcomes"].append(r.outcome)
            if r.outcome_status is not None:
                entry["status"].append(r.outcome_status)

        result: Dict[str, Dict[str, Dict[str, dict]]] = {}
        for trend_type, trend in raw.items():
            result[trend_type] = {}
            total = by_trend_total[trend_type]
            for zone_type, zone in trend.items():
                result[trend_type][zone_type] = {}
                zone_total = by_zone_total[trend_type][zone_type]
                for action_rr, entry in zone.items():
                    action_type, rr = action_rr.split("_")
                    outcomes = entry["outcomes"]
                    outcome_status = entry["status"]
                    result[trend_type][zone_type][f"{action_type}_RR{rr}"] = {
                        "count": entry["count"],
                        "freq_within_zone": entry["count"] / zone_total if zone_total else 0.0,
                        "avg_outcome": (sum(outcomes) / len(outcomes)) if outcomes else 0.0,
                        "outcome_status": dict(sorted(Counter(outcome_status).items())) if outcome_status else None,
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
        for trend_type, trend in detail.items():
            print(f"trend_type={trend_type}")
            for zone_type, action_types in trend.items():
                print(f"zone_type={zone_type}")
                for action_type, stat in action_types.items():
                    avg_out = f"{stat['avg_outcome']:.3f}" if stat["avg_outcome"] is not None else "-"
                    line = (
                        f"  {action_type:<10} count={stat['count']}({stat['freq_within_zone']*100:5.1f}%)  "
                        f"OUTCOME={avg_out:>7}"
                    )
                    outcome_status = stat.get("outcome_status")
                    if outcome_status:
                        outcome_status_str = " ".join(f"{k}:{v}" for k, v in outcome_status.items())
                        line += f"  outcome_status=[{outcome_status_str}]"
                    print(line)

    def save_summary_log(self, filepath: str = "summary.log") -> None:
        """Lưu toàn bộ thông tin thống kê giống print_summary vào file log."""
        n = len(self._records)
        n_wf = sum(1 for r in self._records if r.well_formed)
        n_sem = sum(1 for r in self._records if r.well_formed and r.semantic_passed)

        lines = []
        lines.append("=== StatsCollector summary (task2 — action) ===")
        lines.append(f"n_records = {n}")
        if n:
            lines.append(f"well_form_rate = {n_wf / n * 100:.1f}%")
        if n_wf:
            lines.append(f"semantic_pass_rate (trong số well-formed) = {n_sem / n_wf * 100:.1f}%")

        lines.append("\n-- Chi tiết theo zone -> action (đã pass gate) --")
        detail = self.summary()
        if not detail:
            lines.append("  (chưa có record nào pass gate)")
        
        for trend_type, trend in detail.items():
            lines.append(f"trend_type={trend_type}")
            for zone_type, action_types in trend.items():
                lines.append(f"zone_type={zone_type}")
                for action_type, stat in action_types.items():
                    avg_out = f"{stat['avg_outcome']:.3f}" if stat["avg_outcome"] is not None else "-"
                    line = (
                        f"  {action_type:<10} count={stat['count']}({stat['freq_within_zone']*100:5.1f}%)  "
                        f"OUTCOME={avg_out:>7}"
                    )
                    outcome_status = stat.get("outcome_status")
                    if outcome_status:
                        outcome_status_str = " ".join(f"{k}:{v}" for k, v in outcome_status.items())
                        line += f"  outcome_status=[{outcome_status_str}]"
                    lines.append(line)

        # Ghi danh sách chuỗi vào file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        
        print(f"-> Đã lưu log tóm tắt vào: {filepath}")

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
    
if __name__ == "__main__":
    from tlang import ActionType, ZoneDirection, TrendType
    collector = StatsCollector()
    collector.log(TaskRolloutMeta(
        well_formed=True,
        semantic_passed=True,
        trend_type=TrendType.UP.value,
        zone_type=ZoneDirection.support.value,
        action_type=ActionType.BUY.value,
        outcome=1.0,
        outcome_status="WIN_1R",
        rr=1,
    ))
    collector.log(TaskRolloutMeta(
        well_formed=True,
        semantic_passed=True,
        trend_type=TrendType.UP,
        zone_type=ZoneDirection.support,
        action_type=ActionType.BUY,
        outcome=0.0,
        outcome_status="LOSE",
        rr=1,
    ))
    collector.log(TaskRolloutMeta(
        well_formed=True,
        semantic_passed=True,
        trend_type=TrendType.RANGE,
        zone_type=ZoneDirection.support,
        action_type=ActionType.BUY,
        outcome=0.0,
        outcome_status="LOSE",
        rr=1,
    ))
    collector.log(TaskRolloutMeta(
        well_formed=True,
        semantic_passed=True,
        trend_type=TrendType.RANGE,
        zone_type=ZoneDirection.resistance,
        action_type=ActionType.SELL,
        outcome=0.0,
        outcome_status="LOSE",
        rr=1,
    ))
    collector.print_summary()