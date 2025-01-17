import sqlite3
import pandas as pd

# Input database location
db_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022.db"
# Output file location
output_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022_filtered.csv"

# Batch size (number of rows to process per batch)
batch_size = 10000
offset = 0

# Counter for total matched rows
total_matched = 0

try:
    print("Connecting to the database...")
    conn = sqlite3.connect(db_path)
    print("Database connection successful.")

    first_batch = True  # Flag to handle the output file header

    while True:
        # Load the next batch without filtering
        query = f"SELECT * FROM contributions LIMIT {batch_size} OFFSET {offset}"
        print(f"Fetching rows {offset} to {offset + batch_size}...")
        df = pd.read_sql_query(query, conn)

        # If no more rows, exit the loop
        if df.empty:
            print("No more rows to fetch. Exiting loop.")
            break

        # Apply filters: cycle, date, and contributor.type
        filtered_df = df[
            (df['cycle'] >= 2020) & (df['cycle'] <= 2022) &
            (df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31') &
            (df['contributor.type'] == "I")
        ]

        # Count and accumulate matched rows
        batch_matched = len(filtered_df)
        total_matched += batch_matched

        # Skip writing if no rows match
        if filtered_df.empty:
            print(f"No matching rows in batch {offset} to {offset + batch_size}. Skipping.")
            offset += batch_size
            continue

        # Write filtered rows to the output file
        if first_batch:
            filtered_df.to_csv(output_path, index=False, mode='w')  # Overwrite for first batch
            first_batch = False
        else:
            filtered_df.to_csv(output_path, index=False, mode='a', header=False)  # Append for subsequent batches

        print(f"Processed batch with {batch_matched} matching rows. Total matched so far: {total_matched}")

        # Increment the offset for the next batch
        offset += batch_size

    print(f"Filtered data saved successfully to: {output_path}")
    print(f"Total matched rows: {total_matched}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    conn.close()
    print("Database connection closed.")
