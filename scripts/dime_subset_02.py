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
        # Update the query to filter by cycle=2022, seat LIKE 'federal:%', and contributor.type='I'
        query = f"""
            SELECT *
            FROM contributions
            WHERE cycle = 2022
              AND seat LIKE 'federal:%'
              AND "contributor.type" = 'I'
            LIMIT {batch_size} OFFSET {offset}
        """
        print(f"Fetching rows {offset} to {offset + batch_size}...")
        df = pd.read_sql_query(query, conn)

        # If no more rows match the query, exit the loop
        if df.empty:
            print("No more matching rows to fetch. Exiting loop.")
            break

        # Count and accumulate matched rows
        batch_matched = len(df)
        total_matched += batch_matched

        # Write filtered rows to the output file
        if first_batch:
            df.to_csv(output_path, index=False, mode='w')  # Overwrite for the first batch
            first_batch = False
        else:
            df.to_csv(output_path, index=False, mode='a', header=False)  # Append for subsequent batches

        print(f"Processed batch with {batch_matched} rows. Total matched so far: {total_matched}")

        # Increment the offset for the next batch
        offset += batch_size

    print(f"Filtered data saved successfully to: {output_path}")
    print(f"Total matched rows: {total_matched}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    conn.close()
    print("Database connection closed.")
