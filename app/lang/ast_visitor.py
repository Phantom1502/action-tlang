"""
app/lang/ast_visitor.py — Nơi DUY NHẤT serialize AST (ProgramNode/ChartNode/
CandleNode/ThinkNode/ZoneNode) NGƯỢC LẠI thành text — đúng nghịch đảo của
Parser (Parser: text -> AST; ASTVisitor: AST -> text).

THAY THẾ các hàm render rải rác trước đây (tránh 2 nơi định nghĩa cùng 1
logic rồi lệch nhau khi sửa — bài học đã lặp lại nhiều lần trong project):
    - ZoneGenerator._build_completion_text() / _build_chart_text()
      (app/data_prepare/generator.py) -> dùng ASTVisitor thay thế.

Dùng chung digit_pad qua constructor (từ cfg.base.digit_pad) — KHÔNG
hardcode 4 như bản cũ, để đổi digit_pad không phải sửa nhiều nơi.
"""
from __future__ import annotations
from typing import List

from app.lang.ast_nodes import (
    CandleNode,
    ChartNode,
    ProgramNode,
    ThinkNode,
    ZoneNode,
    ActionNode
)


def _digits(n: int, pad: int) -> List[str]:
    return list(str(n).zfill(pad))


class ASTVisitor:
    def __init__(self, digit_pad: int = 4):
        self.digit_pad = digit_pad

    # ------------------------------------------------------------------
    # visit_* — mỗi hàm chỉ lo render ĐÚNG 1 loại node, không biết gì về
    # node cha/con xung quanh (composable — visit_program gọi lại
    # visit_chart/visit_think, không tự viết lại logic của chúng).
    # ------------------------------------------------------------------
    def visit_program(self, program: ProgramNode) -> str:
        parts: List[str] = []
        if program.chart is not None:
            parts.append(self.visit_chart(program.chart))
        if program.think is not None:
            parts.append(self.visit_think(program.think))
        if program.action is not None:
            parts.append(self.visit_action(program.action))
        return " ".join(parts)

    def visit_chart(self, chart: ChartNode) -> str:
        parts = ["<chart>"]
        for candle in chart.candles:
            parts.append(self.visit_candle(candle))
        parts.append("</chart>")
        return " ".join(parts)

    def visit_candle(self, candle: CandleNode) -> str:
        return f"<O_{candle.open}> <H_{candle.high}> <L_{candle.low}> <C_{candle.close}>"

    def visit_think(self, think: ThinkNode) -> str:
        parts = ["<think>"]

        if think.trend is not None:
            parts.append(f"<trend>{think.trend}</trend>")

        if think.current_price_bin is not None:
            parts.append("<current_price>")
            parts.extend(_digits(think.current_price_bin, self.digit_pad))
            parts.append("</current_price>")

        if think.zone is not None:
            parts.append(self.visit_zone(think.zone))

        parts.append("</think>")
        return " ".join(parts)

    def visit_zone(self, zone: ZoneNode) -> str:
        tag = "zone_support" if zone.direction == "support" else "zone_resistance"
        parts = [f"<{tag}>"]
        parts.extend(_digits(zone.lower_bin, self.digit_pad))
        parts.append(":")
        parts.extend(_digits(zone.upper_bin, self.digit_pad))
        parts.append(f"</{tag}>")
        return " ".join(parts)
    
    def visit_action(self, action: ActionNode) -> str:
        parts = ["<action>", action.action_type]
        if action.sl is not None and action.rr is not None:
            parts += ["SL:", *_digits(action.sl, self.digit_pad), f"<RR_{action.rr}>"]
        parts.append("</action>")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Entry point tiện dụng — dùng khi chỉ cần đúng 1 nửa (vd build_grpo_rows
    # chỉ cần chart làm prompt, không cần think; augment chỉ cần build lại
    # completion sau khi dịch bin, không đụng gì tới chart render riêng).
    # ------------------------------------------------------------------
    def render_chart_block(self, candles: List[CandleNode]) -> str:
        chart = ChartNode(candles=candles)
        return self.visit_chart(chart)

    def build_completion(self, think: ThinkNode) -> str:
        return self.visit_think(think)


if __name__ == "__main__":
    from app.lang.parser import Parser
    from app.config.schema import AppConfig
    from app.config.loader import load_config
    cfg: AppConfig = load_config("configs")

    candle1: CandleNode = CandleNode(open=1, high=2, low=3, close=4)
    candle2: CandleNode = CandleNode(open=5, high=6, low=7, close=8)
    chart: ChartNode = ChartNode(candles=[candle1, candle2])
    think: ThinkNode = ThinkNode(
        trend="UP",
        current_price_bin=8,
        zone=ZoneNode(direction="support", lower_bin=1, upper_bin=2)
    )
    action: ActionNode = ActionNode(action_type="BUY", sl=1, rr=2)
    program: ProgramNode = ProgramNode(chart=chart, think=think, action=action)
    visitor = ASTVisitor(digit_pad=cfg.base.digit_pad)

    output = visitor.visit_program(program)
    print(output)

    parser = Parser.from_text(cfg, output)
    result = parser.parse()
    print(result)
    print("well_formed =", result.is_well_formed())
    for e in result.errors:
        print(f"  [{e.severity}] {e.message}")