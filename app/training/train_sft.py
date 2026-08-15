from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("train_sft")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from app.config import (
    AppConfig,
    TrainingConfig,
    load_config
)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_size", choices=["tiny", "base"], default="tiny")
    p.add_argument("--pretrain_repo", default=None, help="Pretrain repo để train sft, cần có để init sft",)
    p.add_argument("--output_dir", required=True)
    
    p.add_argument("--dataset_name", required=True, help="sullivan1502/tlang-pretrain-ids for pretokenized")
    p.add_argument("--dataset_mode", choices=["on_the_fly", "pre_tokenized"], default="pre_tokenized")
    p.add_argument("--cache_dir", default=None, help="local cache dir for huggingface datasets")
    p.add_argument("--max_length", type=int, default=512, help="khớp MAX_POSITION_EMBEDDINGS")

    p.add_argument("--hf_token", default=None, help="HF Token")
    p.add_argument("--repo_id", required=True, help="Model Repo ID on HF Hub")

    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")
    
    return p
def main(cfg: AppConfig) -> None:
    args = build_arg_parser().parse_args()
    
    if args.model_size not in cfg.models.presets:
        print(
            f"--model_size={args.model_size!r} khong co trong configs/models.yaml. "
            f"Cac preset hop le: {sorted(cfg.models.presets.keys())}"
        )
        raise SystemExit(1)

    import os
    
    from transformers import Trainer, TrainingArguments
    from app.tokenizer import load_tokenizer
    from app.training.common import resolve_resume_checkpoint, print_device_info
    from app.training.data import DataArguments, make_data_module
    from app.training.model import ModelLoader
    
    os.makedirs(args.output_dir, exist_ok=True)
    print_device_info()
        
    push_to_hub = False
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        push_to_hub = True
    else:
        print("Có repo_id nhưng chưa có hf_token — nhớ gọi huggingface_hub.login() thủ công trước khi chạy.")

    # ------------------------------------------------------------
    # Tokenizer — luôn load qua Hub (app/tokenizer/hub.py), KHÔNG build lại
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")
          
    # ------------------------------------------------------------
    # Model — 
    # ------------------------------------------------------------
    if args.pretrain_repo is None:
        print("Chưa cố checkpoint pretrain — yêu cầu set --pretrain_repo.")
        exit(1)
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model_loader = ModelLoader(cfg.models, args.model_size)
    model = model_loader.build_continue_model(resume_checkpoint, args.pretrain_repo)
    
    logger.info(f"model vocab_size = {model.config.vocab_size}")
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError(
            f"model.vocab_size ({model.config.vocab_size}) != tokenizer.vocab_size ({tok.vocab_size}) — "
            f"configs/models.yaml.vocab_size dang lech voi tokenizer that dang dung "
            f"(repo_id={args.repo_id!r}). Sua lai configs/models.yaml hoac kiem tra dung tokenizer_repo."
        )
    
    data_args = DataArguments(
        dataset_name=args.dataset_name,
        dataset_mode=args.dataset_mode,
        max_length=args.max_length,
        cache_dir=args.cache_dir
    )
    data_module = make_data_module(tok, data_args, is_pretrain=False)
    from app.training.data import ActionSFTDataset
    data_module = {
        "train_dataset": ActionSFTDataset(data_module["train_dataset"], cfg),
        "eval_dataset": data_module["eval_dataset"],
        "data_collator": data_module["data_collator"],
    }
    
    train_cfg: TrainingConfig = cfg.training['sft']
    print(f"train_cfg = {train_cfg}")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        max_steps=train_cfg.max_steps,
        logging_steps=train_cfg.logging_steps,
        fp16=args.fp16, # tùy môi trường train, nên args để ngoài
        bf16=not args.fp16,
        push_to_hub=push_to_hub,
        hub_model_id=args.repo_id,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=train_cfg.save_steps,
        report_to=[],
    )
    
    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    if push_to_hub:
        trainer.push_to_hub(commit_message="Final sft checkpoint")
        logger.info(f"Đã push bản final lên: https://huggingface.co/{args.repo_id}")
    else:
        logger.info(f"push_to_hub tắt — checkpoint final chỉ lưu local tại {args.output_dir}")

if __name__ == "__main__":
    cfg : AppConfig = load_config("configs")
    main(cfg)