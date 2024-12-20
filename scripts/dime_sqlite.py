import sqlite3
import pandas as pd

# File paths
csv_file = "C:/Users/spatt/Desktop/finetune_FEC_BLS/data/contribs_2022.csv"
db_file = "C:/Users/spatt/Desktop/finetune_FEC_BLS/data/contribs_2022.db"

# Create SQLite connection
conn = sqlite3.connect(db_file)

# Read CSV in chunks and write to SQLite
chunk_size = 10**6
for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
    # Append each chunk to the database
    chunk.to_sql("contributions", conn, if_exists="append", index=False)

print(f"Data has been loaded into {db_file}")
conn.close()
