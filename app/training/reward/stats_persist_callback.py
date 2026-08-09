import os
import logging

logger = logging.getLogger("app.train.reward.stats_persist_callback")

from transformers import TrainerCallback
from app.config.schema import RoundConfig
from app.training.reward.action_buff_controller import EMABuffController, DEFAULT_BUFF_FILENAME
from app.training.reward.stats_collector import StatsCollector, stats_path_for_rank

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



class StatsPersistCallback(TrainerCallback):
    def __init__(self, buff_controller: EMABuffController, stats_collector: StatsCollector, round_config: RoundConfig, output_dir: str):
        self.buff_controller = buff_controller
        self.stats_collector = stats_collector
        self.round_config = round_config
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.stats_path = stats_path_for_rank(self.output_dir, self.round_config.round_id, rank)

    def on_step_end(self, args, state, control, **kwargs):
        for zone_key in self.round_config.zone_buffs.keys():
            action_counts, total = self.stats_collector.counts_since_step_boundary(
                zone_key, key_fn=lambda r: r.action_type
            )
            self.buff_controller.on_step_end(
                round_config=self.round_config,
                zone_key=zone_key,
                counts=action_counts,
                total=total
            )
        self.stats_collector.mark_step_boundary()

    def on_log(self, args, state, control, **kwargs):
        print("\n=== ACTION BUFF CONTROLLER ===")
        for zone_key, zone_value in self.buff_controller.snapshot().items():
            for action_key, metrics in zone_value.items():
                group = f"{zone_key}:{action_key}"
                print(f"{group}: ema_ratio={metrics['ema_ratio']:.4f}, buff={metrics['buff']:.4f}, prev_error={metrics['prev_error']:.4f}")
    
    def on_save(self, args, state, control, **kwargs):
        n_records = len(self.stats_collector._records)
        print(f"\n=== [step={state.global_step}] Chu kỳ report vừa xong ({n_records} record) ===")
        self.stats_collector.print_summary()
        print(f"zone_buff_controller hiện tại: {self.buff_controller.snapshot()}")

        self.stats_collector.save(self.stats_path)
        self.stats_collector.reset()

        ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self.buff_controller.save(os.path.join(ckpt_dir, DEFAULT_BUFF_FILENAME))
            logger.info(f"Đã lưu zone_buff_state -> {ckpt_dir}/")
        else:
            logger.warning(f"Checkpoint dir {ckpt_dir} chưa tồn tại lúc on_save — bỏ qua lưu state.")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n=== [train_end] Chu kỳ report cuối cùng ===")
        self.stats_collector.print_summary()
        print(f"zone_buff_controller cuối: {self.buff_controller.snapshot()}")
        self.stats_collector.save(self.stats_path)