from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Dict, Literal
from enum import Enum
from collections import defaultdict

from tlang import (
    ProgramNode,
    ActionNode,
    CandleNode,
    ZoneNode,
    SemanticChecker,
    Parser,
    ParseResult,
    SemanticResult,
    ZoneDirection,
    ActionType,
    TrendType,
    find_entry_touch,
    zone_score,
    derive_target,
    find_truly_valid_zones
)
from app.training.reward.stats_collector import StatsCollector, TaskRolloutMeta

DEGENERATE_GROUP_STD_EPS = 1e-6
ZONE_PROBE_SL_BUFFER_BINS = 1

@dataclass
class CommonGateResult:
    """Kết quả gate chung (well-formed + semantic) — DÙNG CHUNG cho mọi
    completion trước khi tính zone_score. `gate_score` là điểm liên tục
    (không nhị phân) để GRPO có gradient mượt ngay cả khi fail gate."""
    program: Optional[ProgramNode]
    well_formed: bool
    semantic_result: Optional[SemanticResult]
    passed: bool                 # well_formed AND semantic_result.passed
    gate_score: float
    
class OutcomeStatus(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    ENTRY_TIMEOUT = "ENTRY_TIMEOUT"     # Lệnh limit không bao giờ chạm tới
    HOLD = "HOLD"
    TIMEOUT = "TIMEOUT"
    INVALID_SETUP = "INVALID_SETUP"   # SL sai khoảng cách/phía zone, hoặc target bị bão hoà bin

@dataclass
class ForwardTestResult:
    status: OutcomeStatus
    r_multiple: float                   # có thể dùng để thống kê
    score: float                        # reward cho model học

def forward_test(
    entry_bin: int,
    sl_bin: int,
    target_bin: int,
    future_candles: List[CandleNode],
    action_type: ActionType,
) -> ForwardTestResult:
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        return ForwardTestResult(OutcomeStatus.INVALID_SETUP, 0.0, 0.0)

    for i, candle in enumerate(future_candles):
        if action_type == ActionType.BUY:
            hit_sl = candle.low <= sl_bin
            hit_tp = candle.high >= target_bin
        else:  # short
            hit_sl = candle.high >= sl_bin
            hit_tp = candle.low <= target_bin

        if hit_sl:
            return ForwardTestResult(OutcomeStatus.LOSS, -1.0, 0.0)
        if hit_tp:
            r_multiple = int(abs(target_bin - entry_bin) / risk)
            if r_multiple == 1: # r_multiple = 1
                return ForwardTestResult(OutcomeStatus.WIN, r_multiple, r_multiple)
            else: # r_multiple > 1
                return ForwardTestResult(OutcomeStatus.WIN, r_multiple, r_multiple)
            
    return ForwardTestResult(OutcomeStatus.TIMEOUT, 0.0, 0.0)

def eval_outcome(
    current_price: int,
    zone: ZoneNode,
    sl: int,
    rr: int,
    future_candles: List[CandleNode],
    action_type: ActionType,
) -> ForwardTestResult:
    entry = None
    if zone.direction == ZoneDirection.support:
        if zone.lower_bin > current_price:
            return ForwardTestResult(OutcomeStatus.INVALID_SETUP, 0.0, 0.0)
        if zone.lower_bin <= current_price <= zone.upper_bin:
            entry = current_price # nếu giá trong zone, vào lệnh ngay tại giá hiện tại
        else:
            entry = zone.upper_bin # nếu giá trên sup zone, đặt limit tại zone upper bin
    else:
        if zone.upper_bin < current_price:
            return ForwardTestResult(OutcomeStatus.INVALID_SETUP, 0.0, 0.0)
        if zone.lower_bin <= current_price <= zone.upper_bin:
            entry = current_price # nếu giá trong zone, vào lệnh ngay tại giá hiện tại
        else:
            entry = zone.lower_bin # nếu giá dưới res zone, đặt limit tại zone lower bin
            
    if entry is None:
        return ForwardTestResult(OutcomeStatus.INVALID_SETUP, 0.0, 0.0)
    
    touch_idx = None
    if zone.direction == ZoneDirection.support:
        touch_idx = find_entry_touch(entry, ActionType.BUY, future_candles)
    else:
        touch_idx = find_entry_touch(entry, ActionType.SELL, future_candles)
        
    if touch_idx is None:
        return ForwardTestResult(OutcomeStatus.ENTRY_TIMEOUT, 0.0, 1.0)

    target_bin = derive_target(entry, sl, rr, zone.direction)

    return forward_test(entry, sl, target_bin, future_candles[touch_idx:], action_type)

class TLangReward:
    def __init__(
        self, 
        cfg: AppConfig,
        stats_collector: Optional[StatsCollector] = None,
        mode="train"
    ):
        self.__name__ = "TLangReward"
        self.cfg = cfg
        self.stats_collector = stats_collector
        self.mode = mode
        
    def common_check(
        self,
        parse_result: ParseResult,
        program: ProgramNode,
    ) -> CommonGateResult:
        if not parse_result.is_well_formed():
            return CommonGateResult(
                program=program,
                well_formed=False,
                semantic_result=None,
                passed=False,
                gate_score=parse_result.well_form_score(),
            )
            
        semantic_result: SemanticResult = SemanticChecker(self.cfg.tlang).check(program)
        if not semantic_result.passed:
            return CommonGateResult(
                program=program,
                well_formed=True,
                semantic_result=semantic_result,
                passed=False,
                gate_score=semantic_result.score,
            )

        return CommonGateResult(
            program=program,
            well_formed=True,
            semantic_result=semantic_result,
            passed=True,
            gate_score=semantic_result.score + parse_result.well_form_score(),
        )
        
    def action_score(
        self,
        program: ProgramNode,
        future_bins: List[CandleNode],
    ) -> ForwardTestResult:
        action: ActionNode = program.action
        if action.action_type == ActionType.HOLD:
            return ForwardTestResult(
                OutcomeStatus.HOLD,
                0.0,
                0.0 # 
            )
        
        return eval_outcome(
            current_price=program.chart.current_price,
            zone=program.think.zone,
            sl=action.sl,
            rr=action.rr,
            future_candles=future_bins,
            action_type=action.action_type
        )
        
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        future_candles: List[CandleNode],
        hints: List[Tuple[str, str]]
    ) -> Tuple[float, TaskRolloutMeta]:
        reward = 0.0

        parse_result: ParseResult = Parser.from_text(self.cfg.tlang, prompt + " " + completion).parse()
        program = parse_result.ast
        common_result: CommonGateResult = self.common_check(parse_result, program)
        reward += common_result.gate_score
        
        if not common_result.passed:
            meta = TaskRolloutMeta(
                well_formed=parse_result.is_well_formed(),
                semantic_passed=False,
                trend_passed=False,
                action_passed=False,
                trend_type=None,
                zone_type=None,
                action_type=None,
                outcome=None,
                outcome_status=None,
                rr=None
            )
            return reward, meta
        
        
        if self.mode == "train":
            # check match
            if program.think.trend.value not in [h[0] for h in hints]:
                return reward, TaskRolloutMeta(
                    well_formed=True,
                    semantic_passed=True,
                    trend_passed=False,
                    action_passed=False,
                    trend_type=None,
                    zone_type=None,
                    action_type=None,
                    outcome=None,
                    outcome_status=None,
                    rr=None
                )
            reward += 0.5
            
            valid_pairs = [(h[0], h[1]) for h in hints]
            pair = (program.think.trend.value, program.action.action_type.value)
            if pair not in valid_pairs:
                return reward, TaskRolloutMeta(
                    well_formed=True,
                    semantic_passed=True,
                    trend_passed=True,
                    action_passed=False,
                    trend_type=None,
                    zone_type=None,
                    action_type=None,
                    outcome=None,
                    outcome_status=None,
                    rr=None
                )
            reward += 0.5
        
        # if all pass, forward test
        score: ForwardTestResult = self.action_score(program, future_candles)
        reward += score.score
        
        meta = TaskRolloutMeta(
            well_formed=True,
            semantic_passed=True,
            trend_passed=True,
            action_passed=True,
            trend_type=program.think.trend.value,
            zone_type=program.think.zone_type.value,
            action_type=program.action.action_type.value,
            outcome=score.r_multiple,
            outcome_status=score.status.value,
            rr=program.action.rr
        )
        return reward, meta
    
    def _caching_hint_type(self, prompt: str, future_candles: List[CandleNode]) -> List[Tuple[str, str]]:
        cached = self._cached_hint_type.get(prompt)
        if cached is not None:
            return cached

        BORDER_LOW, BORDER_HIGH = 0.5, 0.7   # vùng biên quanh trend_threshhold=0.6 -- CHỈ nới trong dải này

        results: List[Tuple[str, str]] = []
        candles: List[CandleNode] = Parser.from_text(self.cfg.tlang, prompt).parse().ast.chart.candles
        for zone_direction in (ZoneDirection.support, ZoneDirection.resistance):
            zones = find_truly_valid_zones(
                candles, 
                self.cfg.tlang.last_n_touch, 
                future_candles,
                zone_direction, 
                swing_window=5, 
                zone_width=self.cfg.tlang.zone_range[0],
                max_bin=self.cfg.tlang.n_bins,
            )
            best_score = None
            best_pairs: List[Tuple[str, str]] = []
            for _, lower_bin, upper_bin, _ in zones:
                zone = ZoneNode(zone_direction, lower_bin, upper_bin)
                score = zone_score(zone, future_candles, self.cfg.base.rr_min, self.cfg.base.rr_max) * self.cfg.base.zone_score_weight

                if score > 0.6:
                    if zone_direction == ZoneDirection.support:
                        pairs = [(TrendType.UP.value, ActionType.BUY.value)]
                    else:
                        pairs = [(TrendType.DOWN.value, ActionType.SELL.value)]
                    # CHỈ nới thêm RANGE khi score nằm sát biên dưới (0.6, 0.7) --
                    # score đã cao hẳn (>0.7) là trend rõ ràng, KHÔNG cho lách qua RANGE.
                    if score <= BORDER_HIGH:
                        if zone_direction == ZoneDirection.support:
                            pairs.append((TrendType.RANGE.value, ActionType.BUY.value))
                        else:
                            pairs.append((TrendType.RANGE.value, ActionType.SELL.value))
                elif score > 0.3:
                    if zone_direction == ZoneDirection.support:
                        pairs = [(TrendType.RANGE.value, ActionType.BUY.value)]
                        # nới NGƯỢC: score sát biên trên (0.5, 0.6] của vùng RANGE cũng cho UP đi kèm
                        if score > BORDER_LOW:
                            pairs.append((TrendType.UP.value, ActionType.BUY.value))
                    else:
                        pairs = [(TrendType.RANGE.value, ActionType.SELL.value)]
                        if score > BORDER_LOW:
                            pairs.append((TrendType.DOWN.value, ActionType.SELL.value))
                else:
                    continue

                if best_score is None or score > best_score:
                    best_score = score
                    best_pairs = pairs

            if best_pairs:
                results.extend(best_pairs)

        if len(results) == 0:
            results.append((TrendType.RANGE.value, ActionType.HOLD.value))

        self._cached_hint_type[prompt] = results
        return results
    
    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[str],
        future_bins: Sequence[Sequence[Sequence[int]]],
        **kwargs,
    ) -> List[float]:
        n = len(prompts)
        rewards: List[float] = [0.0] * n
        metas: List[TaskRolloutMeta] = [None] * n

        self._cached_hint_type: Dict[Any, List[Tuple[str, str]]] = {}
        for i in range(n):
            future_candles: List[CandleNode] = [CandleNode(c[0], c[1], c[2], c[3]) for c in future_bins[i]]
            hints: List[Tuple[str, str]] = self._caching_hint_type(prompts[i], future_candles)
            reward, meta = self.compute_reward(
                prompts[i], completions[i], future_candles, hints,
            )
            rewards[i] = reward
            metas[i] = meta
            
        groups_idx: Dict[Any, List[int]] = defaultdict(list)
        for i, p in enumerate(prompts):
            groups_idx[p].append(i)

        for idx_list in groups_idx.values():
            vals = [rewards[i] for i in idx_list]
            cnt = len(idx_list)
            mean = sum(vals) / cnt
            variance = sum((v - mean) ** 2 for v in vals) / cnt
            std = variance ** 0.5

            too_hard = not any(metas[i].trend_passed for i in idx_list)
            too_easy = std < DEGENERATE_GROUP_STD_EPS

            if too_hard or too_easy:
                for i in idx_list:
                    rewards[i] = mean
            else:
                if self.stats_collector is not None:
                    for i in idx_list:
                        self.stats_collector.log(metas[i])
                
        return rewards
    
if __name__ == "__main__":
    from app.config import load_config, AppConfig
    from tlang import (
        Parser,
        ProgramNode
    )
    
    cfg: AppConfig = load_config("configs")
    
    stat: StatsCollector = StatsCollector()
    
    prompt = "<chart> <O_1024> <H_1059> <L_999> <C_1033> <O_1029> <H_1033> <L_1004> <C_1004> <O_1004> <H_1016> <L_978> <C_992> <O_992> <H_1023> <L_973> <C_1021> <O_1021> <H_1030> <L_984> <C_988> <O_988> <H_992> <L_955> <C_962> <O_962> <H_967> <L_915> <C_938> <O_938> <H_947> <L_926> <C_942> <O_941> <H_976> <L_933> <C_976> <O_975> <H_983> <L_947> <C_951> <O_951> <H_984> <L_950> <C_979> <O_981> <H_998> <L_960> <C_991> <O_990> <H_1003> <L_975> <C_982> <O_982> <H_994> <L_960> <C_992> <O_992> <H_1011> <L_988> <C_1000> <O_1000> <H_1004> <L_963> <C_965> <O_964> <H_978> <L_932> <C_934> <O_935> <H_958> <L_933> <C_944> <O_943> <H_949> <L_910> <C_912> <O_912> <H_944> <L_899> <C_932> <O_932> <H_963> <L_917> <C_962> <O_965> <H_970> <L_914> <C_920> <O_919> <H_944> <L_914> <C_921> <O_919> <H_926> <L_894> <C_897> <O_897> <H_916> <L_892> <C_895> <O_895> <H_907> <L_886> <C_905> <O_905> <H_935> <L_895> <C_926> <O_926> <H_943> <L_923> <C_928> <O_928> <H_959> <L_926> <C_951> <O_950> <H_954> <L_934> <C_937> <O_937> <H_937> <L_898> <C_914> <O_914> <H_945> <L_913> <C_937> <O_938> <H_960> <L_935> <C_954> <O_953> <H_967> <L_930> <C_938> <O_938> <H_944> <L_910> <C_925> <O_925> <H_930> <L_912> <C_915> <O_915> <H_915> <L_864> <C_865> <O_866> <H_891> <L_866> <C_889> <O_889> <H_896> <L_846> <C_851> <O_851> <H_866> <L_842> <C_848> <O_846> <H_862> <L_843> <C_857> <O_858> <H_873> <L_836> <C_872> <O_872> <H_901> <L_872> <C_899> <O_900> <H_925> <L_882> <C_924> <O_924> <H_937> <L_924> <C_935> <O_934> <H_934> <L_907> <C_914> <O_915> <H_926> <L_903> <C_905> <O_906> <H_907> <L_893> <C_898> <O_898> <H_914> <L_891> <C_909> <O_909> <H_911> <L_897> <C_909> <O_909> <H_930> <L_902> <C_929> <O_929> <H_931> <L_904> <C_915> <O_915> <H_928> <L_913> <C_921> <O_922> <H_934> <L_917> <C_926> <O_926> <H_926> <L_905> <C_915> <O_913> <H_920> <L_889> <C_903> <O_901> <H_902> <L_877> <C_881> <O_881> <H_889> <L_866> <C_877> <O_877> <H_887> <L_864> <C_867> <O_868> <H_875> <L_858> <C_869> <O_869> <H_891> <L_867> <C_878> <O_879> <H_882> <L_841> <C_846> <O_847> <H_870> <L_842> <C_863> <O_863> <H_869> <L_849> <C_865> <O_865> <H_876> <L_841> <C_852> <O_852> <H_853> <L_837> <C_840> <O_840> <H_843> <L_815> <C_825> <O_825> <H_834> <L_804> <C_809> <O_811> <H_846> <L_806> <C_843> <O_842> <H_842> <L_816> <C_821> <O_821> <H_861> <L_820> <C_857> <O_857> <H_869> <L_847> <C_869> <O_869> <H_870> <L_824> <C_826> <O_826> <H_836> <L_808> <C_832> <O_832> <H_846> <L_829> <C_838> <O_838> <H_855> <L_838> <C_852> <O_852> <H_872> <L_844> <C_844> <O_845> <H_884> <L_845> <C_879> <O_879> <H_893> <L_876> <C_888> <O_888> <H_889> <L_872> <C_873> <O_873> <H_875> <L_847> <C_856> <O_855> <H_876> <L_854> <C_863> <O_864> <H_864> <L_843> <C_843> <O_843> <H_849> <L_824> <C_824> <O_825> <H_836> <L_780> <C_784> <O_784> <H_841> <L_782> <C_835> <O_836> <H_840> <L_809> <C_811> <O_811> <H_814> <L_796> <C_799> <O_799> <H_816> <L_799> <C_808> <O_807> <H_828> <L_804> <C_824> <O_824> <H_824> <L_811> <C_818> <O_816> <H_828> <L_809> <C_818> <O_818> <H_830> <L_813> <C_814> <O_816> <H_822> <L_802> <C_810> <O_810> <H_816> <L_791> <C_807> <O_808> <H_811> <L_769> <C_771> <O_771> <H_784> <L_751> <C_783> <O_783> <H_804> <L_782> <C_798> <O_798> <H_825> <L_795> <C_822> <O_822> <H_825> <L_807> <C_813> </chart>"
    completion = "<think> <trend>UP</trend> <current_price> 0 8 1 3 </current_price> <zone_support> 0 7 5 9 : 0 8 1 0 </zone_support> </think> <action> BUY SL: 0 7 3 9 <RR_1> </action>" 
    future_bins: List[List[int]] = [
        [813, 818, 799, 799], [799, 810, 793, 797], [797, 818, 795, 813],
        [813, 815, 805, 812], [814, 825, 809, 812], [812, 813, 794, 794],
        [794, 812, 794, 811], [812, 823, 810, 816], [816, 858, 813, 851],
        [850, 852, 824, 829], [829, 867, 820, 836], [836, 867, 836, 866],
        [868, 881, 845, 881], [882, 891, 868, 873], [874, 876, 846, 859],
        [859, 878, 856, 868], [867, 901, 864, 870], [874, 884, 861, 876],
        [876, 893, 874, 881], [880, 886, 873, 883], [883, 894, 838, 840],
        [837, 846, 827, 840], [840, 860, 840, 845], [845, 856, 835, 855],
        [855, 855, 836, 841], [841, 851, 841, 848], [848, 884, 845, 858],
        [858, 859, 842, 851], [852, 853, 843, 849], [849, 879, 845, 879],
        [878, 904, 867, 869], [869, 871, 855, 869], [869, 870, 855, 855],
        [855, 865, 853, 859], [859, 860, 841, 843], [843, 866, 840, 840],
        [841, 850, 829, 831], [831, 835, 824, 830], [830, 844, 827, 844],
        [844, 872, 832, 864], [864, 868, 855, 865], [865, 871, 857, 865],
        [865, 887, 862, 887], [886, 892, 875, 883], [883, 885, 849, 850],
        [850, 867, 849, 862], [862, 868, 856, 857], [858, 864, 849, 850],
        [851, 853, 834, 849], [849, 861, 843, 858], [858, 870, 855, 868],
        [868, 876, 862, 868], [868, 878, 863, 868], [868, 869, 850, 857],
        [857, 864, 852, 854], [854, 859, 845, 858], [860, 864, 851, 852],
        [852, 853, 842, 850], [850, 855, 843, 855], [855, 869, 848, 869],
        [868, 878, 862, 867], [869, 869, 846, 846], [847, 849, 838, 844],
        [843, 843, 829, 838], [838, 844, 802, 809], [809, 815, 796, 801],
        [801, 802, 773, 774], [773, 803, 769, 788], [790, 800, 786, 790],
        [791, 813, 790, 798], [798, 812, 790, 800], [800, 804, 781, 789],
        [788, 800, 786, 794], [795, 802, 791, 792], [792, 801, 789, 794],
        [794, 811, 792, 810], [810, 829, 810, 823], [823, 832, 819, 821],
        [821, 826, 805, 823], [824, 828, 818, 821], [821, 836, 821, 834],
        [834, 841, 822, 825], [825, 828, 813, 825], [825, 828, 813, 817],
        [816, 817, 794, 796], [795, 796, 771, 778], [779, 802, 777, 799],
        [798, 800, 787, 793], [793, 797, 767, 776], [776, 791, 772, 782],
        [782, 795, 777, 792], [792, 804, 785, 793], [793, 798, 788, 794],
        [794, 799, 788, 798], [799, 799, 780, 782], [782, 788, 774, 784],
        [786, 809, 785, 797], [797, 802, 777, 789], [789, 799, 785, 795],
        [796, 808, 790, 797]
    ]
    
    parseResult = Parser.from_text(cfg.tlang, prompt).parse()
    if not parseResult.is_well_formed():
        print(parseResult.ast)
        print(parseResult.errors)
        
    parseResult.ast.future_bins = [CandleNode(open=c[0], high=c[1], low=c[2], close=c[3]) for c in future_bins]
    program: ProgramNode = parseResult.ast
    
    tlang = TLangReward(cfg, stat)
    reward, meta = tlang.compute_reward(prompt, completion, program.future_bins, [])
    stat.log(meta)
    print(f"Reward: {reward}")
    
    stat.print_summary()