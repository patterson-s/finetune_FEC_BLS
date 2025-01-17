import sqlite3
import pandas as pd

# Input database location
db_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022.db"
# Output file location
output_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022_filtered.csv"

# Batch size (number of rows to process per batch)
batch_size = 10000
# Resume from the last processed batch
offset = 3590000

try:
    print("Connecting to the database...")
    conn = sqlite3.connect(db_path)
    print("Database connection successful.")

    print("Resuming batched query execution...")

    while True:
        # Query a batch of data
        query = f"""
            SELECT *
            FROM contributions
            WHERE cycle BETWEEN 2020 AND 2022
            LIMIT {batch_size} OFFSET {offset}
        """
        
        print(f"Fetching rows {offset} to {offset + batch_size}...")
        df = pd.read_sql_query(query, conn)

        # If no more rows are returned, break out of the loop
        if df.empty:
            print("No more rows to fetch. Exiting loop.")
            break

        # Append new rows to the existing file
        df.to_csv(output_path, index=False, mode='a', header=False)
        print(f"Processed batch with {len(df)} rows. Total processed: {offset + len(df)}")

        # Update offset for the next batch
        offset += batch_size

    print(f"Filtered data saved successfully to: {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the database connection
    conn.close()
    print("Database connection closed.")
