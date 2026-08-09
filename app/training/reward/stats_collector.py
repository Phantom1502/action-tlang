from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def stats_path_for_rank(output_dir: str, round_id: str, rank: int) -> str:
    """NGUỒN DUY NHẤT cho quy ước đặt tên file stats — dùng chung bởi cả
    save-side (StatsPersistCallbackV2.on_save/on_train_end) LẪN load-side
    (train_grpo.py lúc resume). KHÔNG định nghĩa lại công thức này ở nơi
    khác — đổi 1 chỗ, mọi nơi ăn theo, tránh lệch tên file giữa lúc lưu và
    lúc load."""
    return os.path.join(output_dir, f"{round_id}_stats_rank{rank}.json")


@dataclass
class TaskRolloutMeta:
    well_formed: bool
    semantic_passed: bool
    zone_type: Optional[str]          # "NO_ZONE" / "SUP_ZONE" / "RES_ZONE" — None nếu chưa pass gate
    action_type: Optional[str]        # "BUY" / "SELL" / "HOLD" — None nếu chưa pass gate
    buff_applied: Optional[float]     # = buff cộng theo [zone_type, action_type], None nếu chưa pass gate hoặc eval mode (buff_controller=None)


class StatsCollector:
    """
    Nguồn DUY NHẤT cho cả report (print_summary(), gọi theo nhịp save_steps)
    LẪN nuôi buff (counts_since_step_boundary(), gọi theo nhịp optimizer
    step) — task1 CHỈ có 1 task (zone), không cần tham số `task_id` như
    StatsCollectorV2 bản v2 (có 2 task zone/action).

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
        """CHỈ đếm record đã pass gate (well_formed + semantic_passed) —
        khớp đúng quy ước "buff chỉ tính sau khi pass gate" đã chốt thiết
        kế, và cũng tránh report bị nhiễu bởi completion hỏng."""
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
        """Dùng để nuôi buff — CHỈ đếm records kể từ watermark step trước."""
        return self._filter_and_count(self._records[self._step_boundary:], zone_type,key_fn)

    def full_history_counts(self, zone_type: str, key_fn) -> Tuple[Dict[str, int], int]:
        """Dùng cho report — đếm TOÀN BỘ records kể từ lần reset() gần nhất."""
        return self._filter_and_count(self._records, zone_type, key_fn)

    def summary(self) -> Dict[str, Dict[str, dict]]:
        """
        Breakdown theo trend -> zone_type, CHỈ tính trên record đã pass gate
        (well_formed + semantic_passed). Không có r_multiple/rr/win_rate như
        v1/v2 (task1 không có outcome thật, chỉ có zone_quality liên tục).

        touch_rate: tỉ lệ is_touched=True trong số record CÓ zone (SUP/RES)
        của đúng (trend, zone_type) đó — None cho NO_ZONE (is_touched luôn
        None ở nhóm này, không có gì để tính tỉ lệ).
        """
        by_trend_total: Dict[str, int] = defaultdict(int)
        raw: Dict[str, Dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "zone_qualities": [], "buffs": [], "touched": []})
        )
        for r in self._records:
            if r.trend is None or not (r.well_formed and r.semantic_passed) or r.zone_type is None:
                continue
            by_trend_total[r.trend] += 1
            entry = raw[r.trend][r.zone_type]
            entry["count"] += 1
            if r.zone_quality is not None:
                entry["zone_qualities"].append(r.zone_quality)
            if r.buff_applied is not None:
                entry["buffs"].append(r.buff_applied)
            if r.is_touched is not None:
                entry["touched"].append(r.is_touched)

        result: Dict[str, Dict[str, dict]] = {}
        for trend, zone_types in raw.items():
            result[trend] = {}
            total = by_trend_total[trend]
            for zone_type, entry in zone_types.items():
                zqs = entry["zone_qualities"]
                buffs = entry["buffs"]
                touched = entry["touched"]
                result[trend][zone_type] = {
                    "count": entry["count"],
                    "freq_within_trend": entry["count"] / total if total else 0.0,
                    "avg_zone_quality": (sum(zqs) / len(zqs)) if zqs else None,
                    "avg_buff": (sum(buffs) / len(buffs)) if buffs else None,
                    "touch_rate": (sum(touched) / len(touched)) if touched else None,
                }
        return result

    def touch_rate_by_zone_type(self) -> Dict[str, Optional[float]]:
        """Tỉ lệ is_touched=True TOÀN BỘ lịch sử (không stratify theo
        trend), CHỈ cho zone_type có is_touched không None (SUP_ZONE,
        RES_ZONE) — NO_ZONE trả None (không áp dụng khái niệm touch)."""
        counts: Dict[str, List[bool]] = defaultdict(list)
        for r in self._records:
            if not (r.well_formed and r.semantic_passed) or r.zone_type is None or r.is_touched is None:
                continue
            counts[r.zone_type].append(r.is_touched)
        return {
            zt: (sum(vals) / len(vals) if vals else None)
            for zt, vals in counts.items()
        }

    def print_summary(self) -> None:
        n = len(self._records)
        n_wf = sum(1 for r in self._records if r.well_formed)
        n_sem = sum(1 for r in self._records if r.well_formed and r.semantic_passed)

        print("=== StatsCollector summary (task1 — zone) ===")
        print(f"n_records = {n}")
        if n:
            print(f"well_form_rate = {n_wf / n * 100:.1f}%")
        if n_wf:
            print(f"semantic_pass_rate (trong số well-formed) = {n_sem / n_wf * 100:.1f}%")

        print("\n-- Chi tiết theo trend -> zone_type (đã pass gate) --")
        detail = self.summary()
        if not detail:
            print("  (chưa có record nào pass gate)")
        for trend, zone_types in detail.items():
            print(f"trend={trend}")
            for zone_type, stat in zone_types.items():
                avg_zq = f"{stat['avg_zone_quality']:.3f}" if stat["avg_zone_quality"] is not None else "-"
                avg_buff = f"{stat['avg_buff']:.3f}" if stat["avg_buff"] is not None else "-"
                touch_rate = f"{stat['touch_rate']*100:.1f}%" if stat["touch_rate"] is not None else "-"
                print(
                    f"  {zone_type:<10} count={stat['count']:<6} freq={stat['freq_within_trend']*100:5.1f}%  "
                    f"avg_zone_quality={avg_zq:>7}  avg_buff={avg_buff:>7}  touch_rate={touch_rate:>6}"
                )

        print("\n-- Zone_type counts (toàn bộ lịch sử từ lần reset gần nhất, đã pass gate) --")
        zone_counts, zone_total = self.full_history_counts(key_fn=lambda r: r.zone_type)
        for zone_type in sorted(zone_counts.keys()):
            n_zt = zone_counts[zone_type]
            ratio = n_zt / zone_total if zone_total else 0.0
            print(f"  {zone_type:<10} count={n_zt:<6} ratio={ratio * 100:5.1f}%")

        print("\n-- Tỉ lệ zone đã CHẠM (is_touched, toàn bộ lịch sử, KHÔNG stratify theo trend) --")
        touch_rates = self.touch_rate_by_zone_type()
        if not touch_rates:
            print("  (chưa có record nào có zone SUP/RES)")
        for zone_type in sorted(touch_rates.keys()):
            rate = touch_rates[zone_type]
            rate_str = f"{rate * 100:.1f}%" if rate is not None else "-"
            print(f"  {zone_type:<10} touch_rate={rate_str}")

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
                d.setdefault("is_touched", None)   # tương thích ngược file stats cũ chưa có field này
                collector.log(TaskRolloutMeta(**d))
        return collector