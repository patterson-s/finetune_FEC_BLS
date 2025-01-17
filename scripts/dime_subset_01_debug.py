import sqlite3
import pandas as pd

# Input database location
db_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022.db"
# Output file location
output_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022_filtered_debug.csv"

try:
    print("Connecting to the database...")
    conn = sqlite3.connect(db_path)
    print("Database connection successful.")

    # SQL query with filters and limit
    query = """
        SELECT *
        FROM contributions
        WHERE cycle BETWEEN 2020 AND 2022
            AND date BETWEEN '2019-01-01' AND '2022-12-31'
            AND "contributor.type" = 'I'
        LIMIT 1000
    """


    print("Fetching the first 1000 filtered rows...")
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No rows matched the filters.")
    else:
        # Save the result to a CSV file
        df.to_csv(output_path, index=False)
        print(f"Filtered data saved successfully to: {output_path}")
        print(f"Total rows fetched: {len(df)}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    conn.close()
    print("Database connection closed.")
