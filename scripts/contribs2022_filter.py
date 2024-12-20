import pandas as pd

# File path to the large CSV file
file_path = "C:/Users/spatt/Desktop/finetune_FEC_BLS/data/contribs_2022.csv"

# Output file for filtered data
output_file = "C:/Users/spatt/Desktop/finetune_FEC_BLS/data/contribs_2020_2022_filtered.csv"

# Define the filtering criteria
start_date = "2020-01-01"
end_date = "2022-12-31"
target_cycles = [2020, 2022]
contributor_type = 'I'

# Create a list to store chunks of filtered data
filtered_data = []

# Read the CSV file in chunks
chunk_size = 10**6  # Adjust chunk size as needed
for chunk in pd.read_csv(file_path, chunksize=chunk_size, parse_dates=['date'], low_memory=False):
    # Filter by date range and contributor type
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')  # Ensure date parsing
    filtered_chunk = chunk[
        (chunk['date'] >= start_date) &
        (chunk['date'] <= end_date) &
        (chunk['cycle'].isin(target_cycles)) &
        (chunk['contributor.type'] == contributor_type)
    ]
    # Append to the list of filtered data
    filtered_data.append(filtered_chunk)

# Concatenate all filtered chunks into a single DataFrame
if filtered_data:
    filtered_df = pd.concat(filtered_data, ignore_index=True)

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
else:
    print("No data matched the filtering criteria.")
