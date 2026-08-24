#!/usr/bin/env bash
set -euo pipefail

# Chạy 1 lần trước, không nằm trong script:
#   huggingface-cli login   # hoặc: export HF_TOKEN=hf_xxx

# --- Tính toán config trước khi chạy (đổi batch/GPU thì PHẢI tính lại max_steps) ---
#   per_device_train_batch_size = 128
#   gradient_accumulation_steps = 32
#   effective_batch_size        = 128 * 32 = 4096 samples/step
#   max_steps                   = 7000
#   total_samples_seen          = 7000 * 4096 ≈ 28.7M samples (~1 epoch trên dataset 30M docs)
#   ETA thực đo                 = ~25.6s/it * 7000 ≈ 49.7h (khớp num_train_epochs=1)
# --dataset_name "sullivan1502/action-pretrain" nhận trực tiếp từ local

python -m app.training.train_pretrain \
    --model_size base \
    --dataset_name "$DATASET" \
    --dataset_mode local \
    --cache_dir "./cache" \
    \
    --output_dir "./output/base_pretrain" \
    --repo_id "sullivan1502/base-action-pretrain" \
    --hf_token "$HF_TOKEN" \
    \
    --fp16 \