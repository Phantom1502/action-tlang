import os
import logging

logger = logging.getLogger("app.train.reward.stats_persist_callback")

from transformers import TrainerCallback
from app.config.schema import RoundConfig
from app.training.reward.stats_collector import StatsCollector, stats_path_for_rank

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



class StatsPersistCallback(TrainerCallback):
    def __init__(
        self, 
        stats_collector: StatsCollector, 
        round_config: RoundConfig, 
        output_dir: str
    ):
        self.stats_collector = stats_collector
        self.round_config = round_config
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.stats_path = stats_path_for_rank(self.output_dir, self.round_config.round_id, rank)

    def on_step_end(self, args, state, control, **kwargs):
        self.stats_collector.mark_step_boundary()   # vẫn giữ để report theo nhịp save_steps không lẫn dữ liệu

    def on_save(self, args, state, control, **kwargs):
        n_records = len(self.stats_collector._records)
        print(f"\n=== [step={state.global_step}] Chu kỳ report vừa xong ({n_records} record) ===")
        self.stats_collector.print_summary()

        self.stats_collector.save(self.stats_path)
        self.stats_collector.reset()

    def on_train_end(self, args, state, control, **kwargs):
        print("\n=== [train_end] Chu kỳ report cuối cùng ===")
        self.stats_collector.print_summary()
        self.stats_collector.save(self.stats_path)