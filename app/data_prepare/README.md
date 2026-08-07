Module Data Prepare: tạo ra file dataset dùng chung cho sft, grpo sẽ xử lí dữ liệu ntn:

1. Đầu vào là 1 file data parquet hoặc dataset from hf có cấu trúc:
    Prompt, zone completion, future_bins, zone_score, zone_type, window_id
2. Nhiệm vụ:
    - Dựng lại cấu trúc prompt + future_bins => chart object
    - Verify is_price_in_zone (extend)
    - Augment + Gen action
    - Đóng gói thành file dataset hoàn chỉnh cho pretrain, sft
3. Output: là 1 file parquet có cấu trúc:
    Prompt (có zone), completion (action), futures_bins, zone_score, zone_type, window_id

    Với dữ liệu này, trong sft sẽ dùng:
    input: prompt + completion (mask prompt) để train sft

    Với grpo:
    prompt để sinh ra action => futures bins để tính reward

    zone_score để ước lượng chất lượng action so với chất lượng của zone tìm được