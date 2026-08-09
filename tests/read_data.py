from app.config import load_config, AppConfig, RoundConfig, ZoneBuffConfig, ActionBuffConfig
from app.training.reward.action_buff_controller import EMABuffController
from app.training.reward.stats_collector import StatsCollector, TaskRolloutMeta

cfg: AppConfig = load_config("configs")
round_config: RoundConfig = cfg.rounds["round1"]

buff_controller = EMABuffController.load_or_init(
    round_config, 
    resume_checkpoint="."
)

print(buff_controller.states)

stat_collector = StatsCollector()

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="support",
    action_type="BUY",
    buff_applied=True,
)
stat_collector.log(meta)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="support",
    action_type="HOLD",
    buff_applied=True,
)
stat_collector.log(meta)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="resistance",
    action_type="SELL",
    buff_applied=True,
)
stat_collector.log(meta)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="resistance",
    action_type="HOLD",
    buff_applied=True,
)
stat_collector.log(meta)

for zone_key in round_config.zone_buffs.keys():
    action_counts, total = stat_collector.counts_since_step_boundary(
        zone_key, key_fn=lambda r: r.action_type
    )
    buff_controller.on_step_end(
        round_config=round_config,
        zone_key=zone_key,
        counts=action_counts,
        total=total
    )
    
print(buff_controller.get_buff("support", "BUY"))
print(buff_controller.get_buff("support", "HOLD"))
print(buff_controller.get_buff("resistance", "SELL"))
print(buff_controller.get_buff("resistance", "HOLD"))

print(buff_controller.states)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="support",
    action_type="BUY",
    buff_applied=True,
)
stat_collector.log(meta)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="support",
    action_type="HOLD",
    buff_applied=True,
)
stat_collector.log(meta)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="resistance",
    action_type="SELL",
    buff_applied=True,
)
stat_collector.log(meta)

meta = TaskRolloutMeta(
    well_formed=True,
    semantic_passed=True,
    zone_type="resistance",
    action_type="HOLD",
    buff_applied=True,
)
stat_collector.log(meta)

for zone_key in round_config.zone_buffs.keys():
    action_counts, total = stat_collector.counts_since_step_boundary(
        zone_key, key_fn=lambda r: r.action_type
    )
    buff_controller.on_step_end(
        round_config=round_config,
        zone_key=zone_key,
        counts=action_counts,
        total=total
    )
    
print(buff_controller.get_buff("support", "BUY"))
print(buff_controller.get_buff("support", "HOLD"))
print(buff_controller.get_buff("resistance", "SELL"))
print(buff_controller.get_buff("resistance", "HOLD"))

print(buff_controller.states)