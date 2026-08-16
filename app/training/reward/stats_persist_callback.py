import os
import logging

logger = logging.getLogger("app.train.reward.stats_persist_callback")

from transformers import TrainerCallback
from app.config.schema import RoundConfig
from app.training.reward.entropy_controller import EntropyController, DEFAULT_R_ENTROPY_FILENAME, DEFAULT_ENTROPY_FILENAME
from app.training.reward.stats_collector import StatsCollector, stats_path_for_rank

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



class StatsPersistCallback(TrainerCallback):
    def __init__(
        self, 
        entropy_controller: EntropyController, 
        entropy_r_controller: EntropyController,
        stats_collector: StatsCollector, 
        round_config: RoundConfig, 
        output_dir: str
    ):
        self.entropy_controller = entropy_controller
        self.entropy_r_controller = entropy_r_controller
        self.stats_collector = stats_collector
        self.round_config = round_config
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.stats_path = stats_path_for_rank(self.output_dir, self.round_config.round_id, rank)

    def on_step_end(self, args, state, control, **kwargs):
        # KHÔNG còn đếm counts từ stats_collector — entropy đã được record_entropy()
        # trực tiếp trong TLangReward.__call__() (per rollout group), ở đây chỉ flush.
        self.entropy_controller.on_step_end(entropy_config=self.round_config.entropys['completions_entropy'])
        self.entropy_r_controller.on_step_end(entropy_config=self.round_config.entropys['r_entropy'])
        self.stats_collector.mark_step_boundary()   # vẫn giữ để report theo nhịp save_steps không lẫn dữ liệu

    def on_log(self, args, state, control, **kwargs):
        snap = self.entropy_r_controller.snapshot()
        if snap:
            print(
                f"\n=== ENTROPY RR CONTROLLER === \n"
                f"ema_entropy={snap['ema_entropy']:.4f}, bonus={snap['bonus']:.4f}, prev_error={snap['prev_error']:.4f}"
            )
            
        completions_snap = self.entropy_controller.snapshot()
        if completions_snap:
            print(
                f"\n=== ENTROPY COMPLETIONS CONTROLLER === \n"
                f"ema_entropy={completions_snap['ema_entropy']:.4f}, bonus={completions_snap['bonus']:.4f}, prev_error={completions_snap['prev_error']:.4f}"
            )

    def on_save(self, args, state, control, **kwargs):
        n_records = len(self.stats_collector._records)
        print(f"\n=== [step={state.global_step}] Chu kỳ report vừa xong ({n_records} record) ===")
        self.stats_collector.print_summary()
        print(f"entropy_r_controller hiện tại: {self.entropy_r_controller.snapshot()}")
        print(f"entropy_controller hiện tại: {self.entropy_controller.snapshot()}")

        self.stats_collector.save(self.stats_path)
        self.stats_collector.reset()

        ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self.entropy_r_controller.save(os.path.join(ckpt_dir, DEFAULT_R_ENTROPY_FILENAME))
            logger.info(f"Đã lưu entropy_r_state -> {ckpt_dir}/")
            
            self.entropy_controller.save(os.path.join(ckpt_dir, DEFAULT_ENTROPY_FILENAME))
            logger.info(f"Đã lưu entropy_state -> {ckpt_dir}/")
        else:
            logger.warning(f"Checkpoint dir {ckpt_dir} chưa tồn tại lúc on_save — bỏ qua lưu state.")

    def on_train_end(self, args, state, control, **kwargs):
        print("\n=== [train_end] Chu kỳ report cuối cùng ===")
        self.stats_collector.print_summary()
        print(f"entropy_r_controller cuối: {self.entropy_r_controller.snapshot()}")
        print(f"entropy_controller cuối: {self.entropy_controller.snapshot()}")
        self.stats_collector.save(self.stats_path)