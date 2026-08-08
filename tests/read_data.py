path = r"train_llm.parquet"

import pandas as pd

df = pd.read_parquet(path)

print(df.shape)

print(df["prompt"].iloc[32])
print(df["completion"].iloc[32])
print(df["future_bins"].iloc[32])
print(df["symbol"].iloc[32])
print(df["zone_score"].iloc[32])
print(df["window_id"].iloc[32])