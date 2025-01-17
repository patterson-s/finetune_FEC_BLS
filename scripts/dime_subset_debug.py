import sqlite3

db_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get column names from the contributions table
cursor.execute("PRAGMA table_info(contributions);")
columns = cursor.fetchall()

print("Columns in the contributions table:")
for col in columns:
    print(col[1])  # Print only the column names

conn.close()
