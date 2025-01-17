import sqlite3
import pandas as pd
from tqdm import tqdm

# Input database location
db_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022.db"
# Output file location
output_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022_filtered.csv"

try:
    print("Connecting to the database...")
    conn = sqlite3.connect(db_path)
    print("Database connection successful.")

    # Step 1: Get total count of rows matching the WHERE clause
    count_query = """
        SELECT COUNT(*) as row_count
        FROM contributions
        WHERE cycle = 2022
          AND seat LIKE 'federal:%'
          AND "contributor.type" = 'I'
    """
    cursor = conn.cursor()
    cursor.execute(count_query)
    total_rows = cursor.fetchone()[0]
    print(f"Total matching rows: {total_rows}")

    # Step 2: Query for the actual data in chunks
    data_query = """
        SELECT *
        FROM contributions
        WHERE cycle = 2022
          AND seat LIKE 'federal:%'
          AND "contributor.type" = 'I'
    """

    # Choose whatever chunk size you'd like:
    chunk_size = 100000

    # read_sql_query will return an iterator of DataFrames when chunksize is set
    df_iter = pd.read_sql_query(data_query, conn, chunksize=chunk_size)

    # We'll track the number of rows processed
    rows_processed = 0
    first_chunk = True

    print("Beginning chunked read from the database...")

    # Step 3: Wrap df_iter with tqdm for a progress bar.
    # total=ceil(total_rows / chunk_size) to give tqdm an estimate of how many chunks there are:
    from math import ceil
    total_chunks = ceil(total_rows / chunk_size)

    for chunk_num, chunk in enumerate(tqdm(df_iter, total=total_chunks, desc="Reading Chunks"), start=1):
        # Keep track of how many rows we've processed in total
        rows_in_chunk = len(chunk)
        rows_processed += rows_in_chunk

        # Write to CSV
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)

        # Optional: print or log incremental progress
        print(f"Chunk {chunk_num} done: {rows_in_chunk} rows in this chunk. "
              f"Total processed so far: {rows_processed}/{total_rows}.")

    print(f"Finished reading in chunks. Total rows written: {rows_processed}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    conn.close()
    print("Database connection closed.")
