from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Dict
from enum import Enum

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
)
from app.training.reward.stats_collector import StatsCollector, TaskRolloutMeta

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

def _find_entry_touch(entry_price: int, type: ActionType, candles: List[CandleNode]) -> Optional[int]:
    """Index nến ĐẦU TIÊN có [low,high] giao với [zone.lower_bin,
    zone.upper_bin] — None nếu không nến nào chạm trong toàn bộ `candles`
    (caller đã cắt đúng outcome_horizon trước khi truyền vào)."""
    for i, c in enumerate(candles):
        if type == ActionType.BUY:
            if c.low <= entry_price:
                return i
        else:
            if c.high >= entry_price:
                return i
    return None

def derive_target(
    entry_bin: int, 
    sl_bin: int, 
    rr: int, 
    direction: str,
) -> Optional[int]:
    if direction == "long":
        target = entry_bin + rr * (entry_bin - sl_bin)
    else:
        target = entry_bin - rr * (sl_bin - entry_bin)
    return round(target)

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
                return ForwardTestResult(OutcomeStatus.WIN, r_multiple, 0.5)
            else: # r_multiple > 1
                return ForwardTestResult(OutcomeStatus.WIN, r_multiple, 1.5)
            
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
        touch_idx = _find_entry_touch(entry, ActionType.BUY, future_candles)
    else:
        touch_idx = _find_entry_touch(entry, ActionType.SELL, future_candles)
        
    if touch_idx is None:
        return ForwardTestResult(OutcomeStatus.ENTRY_TIMEOUT, 0.0, 1.0)

    target_bin = derive_target(entry, sl, rr, zone.direction)

    return forward_test(entry, sl, target_bin, future_candles[touch_idx:], action_type)

class TLangReward:
    def __init__(
        self, 
        cfg: AppConfig,
        stats_collector: Optional[StatsCollector] = None
    ):
        self.__name__ = "TLangReward"
        self.cfg = cfg
        self.stats_collector = stats_collector
        
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
        '''
        Phần thưởng sẽ có 4 mốc:
        - 0.0: lệnh lỗi, bị dính sl, timeout
        - 0.5: rr = 1
        - 1.0: HOLD, entry timeout (ko vào đc lệnh thì ko lỗ, tương đương HOLD)
        - 1.5: lệnh thắng với RR >= 2
        '''
        
        action: ActionNode = program.action
        if action.action_type == ActionType.HOLD:
            return ForwardTestResult(
                OutcomeStatus.HOLD,
                0.0,
                1.0 # 
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
                trend_type=None,
                zone_type=None,
                action_type=None,
                outcome=None,
                outcome_status=None,
                rr=None
            )
            return reward, meta
        
        score: ForwardTestResult = self.action_score(program, future_candles)
        reward += score.score
        
        meta = TaskRolloutMeta(
            well_formed=True,
            semantic_passed=True,
            trend_type=program.think.trend.value,
            zone_type=program.think.zone_type.value,
            action_type=program.action.action_type.value,
            outcome=score.r_multiple,
            outcome_status=score.status.value,
            rr=program.action.rr
        )
        return reward, meta
    
    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[str],
        future_bins: Sequence[Sequence[Sequence[int]]],
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        for prompt, completion, future_bin in zip(prompts, completions, future_bins):
            future_candles = [CandleNode(o[0], o[1], o[2], o[3]) for o in future_bin]
            reward, meta = self.compute_reward(prompt, completion, future_candles)
            rewards.append(reward)
                
            if self.stats_collector is not None:
                self.stats_collector.log(meta)      
                
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
    
    parseResult = Parser.from_text(cfg.tlang, prompt + " " + completion).parse()
    if not parseResult.is_well_formed():
        print(parseResult.errors)
        
    parseResult.ast.future_bins = [CandleNode(open=c[0], high=c[1], low=c[2], close=c[3]) for c in future_bins]
    program: ProgramNode = parseResult.ast
    
    tlang = TLangReward(cfg, stat)
    reward, meta = tlang.compute_reward(prompt, completion, program.future_bins)
    stat.log(meta)
    print(f"Reward: {reward}")
    
    stat.print_summary()