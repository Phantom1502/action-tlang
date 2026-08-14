import duckdb

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
        input_path='data/raw/last-checkpoint-2500/train/*.parquet',
        output_path='data/filter/train-checkpoint-2500.parquet'
    )

def filter_val_ds():
    filter_price_in_zone(
        input_path='data/raw/last-checkpoint-2500/val/*.parquet',
        output_path='data/filter/val-checkpoint-2500.parquet'
    )
    
if __name__ == '__main__':
    filter_train_ds()
    filter_val_ds()
    pass