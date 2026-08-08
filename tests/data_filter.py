import duckdb

# DuckDB sẽ tự quét tất cả file shard, lọc theo block và ghi thẳng ra file mới
query = """
    COPY (
        SELECT * 
        FROM 'data/raw/val/*.parquet' 
        WHERE price_in_zone_now = True
    ) 
    TO 'data/filter/val.parquet' (FORMAT PARQUET);
"""

duckdb.execute(query)

def filter_price_in_zone(
    input_path: str,
    output_path: str
):
    query = f"""
        COPY (
            SELECT * 
            FROM '{input_path}' 
            WHERE price_in_zone_now = True
        ) 
        TO '{output_path}' (FORMAT PARQUET);
    """
    duckdb.execute(query)

def filter_train_ds():
    filter_price_in_zone(
        input_path='data/raw/train/*.parquet',
        output_path='data/filter/train.parquet'
    )

def filter_val_ds():
    filter_price_in_zone(
        input_path='data/raw/val/*.parquet',
        output_path='data/filter/val.parquet'
    )
    
if __name__ == '__main__':
    filter_train_ds()
    filter_val_ds()
    pass