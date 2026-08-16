from app.config import load_config, AppConfig
from app.training.reward.entropy_controller import EntropyController

if __name__ == "__main__":
    cfg : AppConfig = load_config("configs")

    entropy: EntropyController = EntropyController.load_or_init(
        cfg.rounds['round1'].entropys['completions_entropy'], 
        file_name="entropy_state.json",
        resume_checkpoint="checkpoints/1/entropy_state.json"
    )