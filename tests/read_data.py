file = r"data\dataset\val_llm.parquet"

import pandas as pd

df = pd.read_parquet(file)
# list all columns
print(df.columns)

# total records
print(df.shape[0])

# count well-formed records
print(df["well_formed"].sum())

# count semantic-passed records
print(df["semantic_passed"].sum())

# cal mean of zone_quality
print(df["zone_quality"].mean())
# list all zone type
print(df["zone_type"].unique())
# count all zone type = SUP_ZONE
print(df["zone_type"].value_counts())