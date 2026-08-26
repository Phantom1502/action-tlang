#!/usr/bin/env bash
set -euo pipefail

# Chạy 1 lần trước, không nằm trong script:
#   huggingface-cli login   # hoặc: export HF_TOKEN=hf_xxx

# --- Tính toán config trước khi chạy (đổi batch/GPU thì PHẢI tính lại max_steps) ---
#   per_device_train_batch_size = 16
#   gradient_accumulation_steps = 16
#   effective_batch_size        = 16 * 16 = 256 samples/step
#   max_steps                   = 7000
#   total_samples_seen          = 7000 * 256 ≈ 28.7M samples (~1 epoch trên dataset 30M docs)
#   ETA thực đo                 = ~25.6s/it * 7000 ≈ 49.7h (khớp num_train_epochs=1)


python -m app.training.train_grpo \
    --round_id round1 \
    --model_size base \
    --source_repo "sullivan1502/base-action-pretrain" \
    --dataset_name "sullivan1502/action-grpo-data" \
    \
    --max_completion_length 32 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --num_generations 16 \
    \
    --output_dir "./output/base_grpo" \
    --repo_id "sullivan1502/base-action-grpo-round1" \
    --hf_token "$HF_TOKEN" \
    \
    --fp16 \