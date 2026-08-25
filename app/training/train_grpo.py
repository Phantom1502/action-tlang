from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("train_grpo")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from app.config import (
    AppConfig, 
    RoundConfig,
    TrainingConfig,
    load_config
)

def _seed_from_round_id(round_id: str) -> int:
    import hashlib
    return int(hashlib.md5(round_id.encode()).hexdigest(), 16) % (2**31)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument(
        "--source_repo", default=None,
        help="Nguồn init NẾU round này chưa có checkpoint nào: SFT repo (round 1) "
             "hoặc checkpoint round liền trước (round N>1) — truyền tay, không tự động.",
    )
    p.add_argument("--output_dir", required=True)
    
    p.add_argument("--dataset_name", default=None, help="Dataset GRPO gốc (schema prompt/future_bins/symbol/window_id)")
    p.add_argument("--train_split", default="train")
    p.add_argument("--round_id", required=True, help="round id, vd: round1")

    p.add_argument("--max_completion_length", type=int, default=64)

    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    
    p.add_argument("--num_generations", type=int, default=12)
    p.add_argument("--use_vllm", action="store_true", default=False)
    
    p.add_argument("--hf_token", default=None, help="HF Token")
    p.add_argument("--repo_id", required=True, help="Model Repo ID on HF Hub")

    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")
    
    return p

def main(cfg: AppConfig):
    args = build_arg_parser().parse_args()
    
    if args.model_size not in cfg.models.presets:
        print(
            f"--model_size={args.model_size!r} khong co trong configs/models.yaml. "
            f"Cac preset hop le: {sorted(cfg.models.presets.keys())}"
        )
        raise SystemExit(1)
    
    if not args.repo_id or not args.dataset_name:
        print("--repo_id và --dataset_name bắt buộc khi train thật.")
        raise SystemExit(1)
    
    from app.training.common import resolve_resume_checkpoint, print_device_info
    from app.tokenizer import load_tokenizer
    from app.training.model import ModelLoader
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    from app.training.reward import (
        TLangReward, 
        StatsCollector, 
        EntropyController, 
        DEFAULT_ENTROPY_FILENAME,
        DEFAULT_R_ENTROPY_FILENAME,
        StatsPersistCallback,
        stats_path_for_rank
    )
        
    os.makedirs(args.output_dir, exist_ok=True)
    print_device_info()
    
    push_to_hub = False
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        push_to_hub = True
    else:
        print("Có repo_id nhưng chưa có hf_token — nhớ gọi huggingface_hub.login() thủ công trước khi chạy.")

    # TODO: Init Round here
    round_config: RoundConfig = cfg.rounds[args.round_id]
    print(round_config)
        
    # ------------------------------------------------------------
    # Tokenizer — giống hệt v1 (xem giải thích add_eos_token/add_bos_token
    # quirk trong train_grpo.py v1, không lặp lại).
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")
    tok.add_eos_token = False
    tok.add_bos_token = True
    
    # ------------------------------------------------------------
    # Model — 
    # ------------------------------------------------------------
    if args.source_repo is None:
        print("Chưa cố source_repo — yêu cầu set --source_repo.")
        exit(1)
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model_loader = ModelLoader(cfg.models, args.model_size)
    model = model_loader.build_continue_model(resume_checkpoint, args.source_repo)
    model.config.use_cache = True
    
    logger.info(f"model vocab_size = {model.config.vocab_size}")
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError(
            f"model.vocab_size ({model.config.vocab_size}) != tokenizer.vocab_size ({tok.vocab_size}) — "
            f"configs/models.yaml.vocab_size dang lech voi tokenizer that dang dung "
            f"(repo_id={args.repo_id!r}). Sua lai configs/models.yaml hoac kiem tra dung tokenizer_repo."
        )
    
    # ------------------------------------------------------------
    # StatsCollector — load lại records đã dump của round này (nếu Colab
    # bị ngắt và đây là lần chạy lại) TRƯỚC khi log tiếp.
    # ------------------------------------------------------------
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    stats_path = stats_path_for_rank(args.output_dir, args.round_id, rank=rank)
    stats_collector = StatsCollector.load(stats_path)
    logger.info(f"[rank={rank}] StatsCollector: nạp lại {len(stats_collector._records)} record cũ.")
    
    entropy_r_controller: EntropyController = EntropyController.load_or_init(round_config.entropys["r_entropy"], DEFAULT_R_ENTROPY_FILENAME,resume_checkpoint)
    entropy_controller: EntropyController = EntropyController.load_or_init(round_config.entropys["completions_entropy"], DEFAULT_ENTROPY_FILENAME,resume_checkpoint)
    
    # ------------------------------------------------------------
    # Dataset — load raw GRPO gốc rồi nhân đôi theo task_id (xem cảnh báo
    # ở docstring module về rủi ro group-by-prompt của GRPOTrainer).
    # remove_unused_columns PHẢI False (cần cả future_bins lẫn task_id).
    # ------------------------------------------------------------
    #raw = load_dataset(args.dataset_name, split=args.train_split)
    raw = load_dataset("parquet", data_files=args.dataset_name, split=args.train_split)
    
    train_cfg: TrainingConfig = cfg.training['grpo']
    print(train_cfg)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=_seed_from_round_id(args.round_id),
        remove_unused_columns=False,
        per_device_train_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        max_steps=train_cfg.max_steps,
        logging_steps=train_cfg.logging_steps,
        max_completion_length=args.max_completion_length,

        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        min_p=args.min_p if args.min_p > 0 else None,
        repetition_penalty=args.repetition_penalty,

        num_generations=args.num_generations,
        use_vllm=args.use_vllm,
        fp16=args.fp16,
        bf16=not args.fp16,
        push_to_hub=push_to_hub,
        hub_model_id=args.repo_id if push_to_hub else None,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        save_total_limit=2,
        report_to=[],

        # =====================================================================
        # THÊM MỚI — phanh trực tiếp nhất, rẻ nhất. Log cũ cho thấy grad_norm
        # 2.5-3.9 mà không có max_grad_norm tường minh nào chặn lại (mặc định
        # HF Trainer là 1.0, NHƯNG cần set rõ ràng ở đây để chắc chắn, không
        # phụ thuộc default có thể đổi giữa các version). Nếu sau khi set vẫn
        # thấy log in ra > giá trị này, đó là log đang in grad_norm TRƯỚC khi
        # clip — hành vi bình thường, không phải bug.
        # =====================================================================
        max_grad_norm=1.0,

        # =====================================================================
        # THÊM MỚI — KL penalty với ref model = bản đóng băng của chính
        # checkpoint init (SFT/round trước). Đây LÀ trust-region, KHÔNG liên
        # quan gì tới việc data là gen hay real — chỉ neo policy không trôi
        # quá xa 1 điểm đã biết well-formed/semantic pass. beta nhỏ (0.02-0.05)
        # để không cản học nhưng vẫn có phanh. Bắt đầu 0.02, tăng lên nếu vẫn
        # thấy grad_norm/entropy collapse sau khi áp dụng.
        # =====================================================================
        beta=0.02,

        # =====================================================================
        # THÊM MỚI — chuẩn hoá reward theo std TOÀN BATCH thay vì std của từng
        # nhóm 16 completion/prompt. Với reward hiện tại (2 điểm well-formed/
        # semantic gần như bão hoà + zone_score dao động nhẹ), std TRONG NHÓM
        # rất dễ nhỏ -> advantage bị khuếch đại quá mức cho vài sample lệch
        # nhẹ. "batch" ổn định hơn nhiều vì mẫu số lớn hơn hẳn (batch_size *
        # num_generations thay vì chỉ num_generations).
        # =====================================================================
        scale_rewards="batch",

        # =====================================================================
        # THÊM MỚI — kiểm tra lại cách chuẩn hoá loss theo độ dài completion.
        # "dr_grpo" (Dr.GRPO) chuẩn hoá theo max_completion_length CỐ ĐỊNH thay
        # vì per-sequence-length động (loss_type mặc định "grpo" có thể khuếch
        # đại gradient khi completion ngắn/dao động độ dài như hiện tại,
        # mean_length ~20 nhưng vẫn dao động 10-23). Thử "dr_grpo" trước, nếu
        # không cải thiện thì quay lại "grpo" mặc định.
        # =====================================================================
        loss_type="dr_grpo",

        # =====================================================================
        # THÊM MỚI — set sàn tối thiểu cho std khi chia advantage, phòng
        # trường hợp std khác 0 nhưng RẤT nhỏ (không bị lọc bởi
        # frac_reward_zero_std vì != 0, nhưng vẫn đủ nhỏ để khuếch đại
        # advantage quá mức). Nếu bản trl đang dùng không có tham số này, bỏ
        # dòng này đi — không phải version nào cũng hỗ trợ, kiểm tra
        # GRPOConfig signature trước khi bật.
        # =====================================================================
        # epsilon (nếu trl hỗ trợ) — để mặc định nếu không chắc, không tự chế thêm field.
    )
    
    stats_persist_callback = StatsPersistCallback(
        entropy_controller,
        entropy_r_controller,
        stats_collector, 
        round_config, 
        args.output_dir
    )
    
    tlang_reward = TLangReward(
        cfg,
        entropy_controller,
        entropy_r_controller,
        stats_collector=stats_collector,
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=tlang_reward,
        args=training_args,
        train_dataset=raw,
        processing_class=tok,
        callbacks=[stats_persist_callback],
    )
    
    if training_args.beta != 0.0:
        if not args.source_repo:
            raise RuntimeError(
                "beta != 0 nhưng thiếu --source_repo — không có nguồn cố định để pin ref_model. "
                "Round GRPO luôn cần source_repo (SFT hoặc round trước) làm neo KL cho CẢ round, "
                "kể cả khi model chính đang resume từ checkpoint khác."
            )
        logger.info(f"Pin ref_model vào nguồn init cố định của round: {args.source_repo} (KHÔNG đổi theo resume)")
        ref_model_loader = ModelLoader(cfg.models, args.model_size)
        ref_model = ref_model_loader.load_ref_model(args.source_repo)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.to(model.device)
        trainer.ref_model = ref_model
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    trainer.save_model()
    canonical_tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    canonical_tok.save_pretrained(args.output_dir)
    entropy_controller.save(os.path.join(args.output_dir, DEFAULT_ENTROPY_FILENAME))
    entropy_r_controller.save(os.path.join(args.output_dir, DEFAULT_R_ENTROPY_FILENAME))
    stats_collector.save(stats_path)

    if push_to_hub:
        trainer.push_to_hub(commit_message=f"GRPO v2 {args.round_id} checkpoint")
        logger.info(f"Đã push lên: https://huggingface.co/{args.repo_id}")

    if trainer.is_world_process_zero():
        print(f"\n=== Report round {args.round_id} (rank {rank} — chạy lại với --report_only để gộp mọi rank) ===")
        stats_collector.print_summary()
        
if __name__ == "__main__":
    cfg : AppConfig = load_config("configs")
    main(cfg)