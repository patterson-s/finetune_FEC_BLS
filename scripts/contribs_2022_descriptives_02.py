import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_finetune_decoder(decoder_path: str):
    """Load and preprocess the decoder map from a JSONL file."""
    decoder_map = {}
    with open(decoder_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract job title from prompt
            prompt = entry['prompt']
            if "classify this profession:" in prompt:
                job_title = prompt.split(":")[1].split("->")[0].strip()
                decoder_map[job_title] = entry['completion']
    return decoder_map

def decode_insufficient_info(merged_df: pd.DataFrame, original_df: pd.DataFrame, decoder_map: Dict[str, str]) -> int:
    """
    Attempt to decode rows with 'insufficient_information_gpt' values using the original dataset and decoder map.
    Returns the count of successfully decoded rows.
    """
    updated_count = 0

    # Iterate over rows where 'decoded_classification' is 'insufficient_information_gpt'
    for idx in merged_df.index[merged_df['decoded_classification'] == "insufficient_information_gpt"]:
        try:
            # Get the corresponding occupation from the original dataset
            original_occupation = original_df.loc[idx, 'most.recent.contributor.occupation']

            # Check if the original occupation exists in the decoder map
            if pd.notna(original_occupation) and original_occupation in decoder_map:
                # Replace 'insufficient_information_gpt' with the mapped value
                merged_df.at[idx, 'decoded_classification'] = decoder_map[original_occupation]
                updated_count += 1
        except KeyError:
            # Log a warning if the index is not found in the original dataset
            logger.warning(f"Index {idx} not found in original dataset.")

    logger.info(f"Successfully decoded {updated_count} rows.")
    return updated_count



def descriptive_statistics(original_path: str, merged_path: str, decoder_path: str):
    """Generate descriptive statistics and attempt to reduce 'insufficient_information_gpt'."""
    
    # Load datasets
    logger.info("Loading the original dataset...")
    original_df = pd.read_csv(original_path)
    
    logger.info("Loading the merged dataset...")
    merged_df = pd.read_csv(merged_path)
    
    logger.info("Loading the finetune decoder file...")
    decoder_map = load_finetune_decoder(decoder_path)
    
    # 1. Check if the merged dataset has the same number of rows as the original
    logger.info("Comparing row counts...")
    original_rows = len(original_df)
    merged_rows = len(merged_df)
    logger.info(f"Original dataset rows: {original_rows}")
    logger.info(f"Merged dataset rows: {merged_rows}")
    
    if original_rows == merged_rows:
        logger.info("The merged dataset contains the same number of rows as the original dataset.")
    else:
        logger.warning("The merged dataset does NOT contain the same number of rows as the original dataset.")
    
    # 2. NA values in 'most.recent.contributor.occupation'
    logger.info("Calculating NA values in 'most.recent.contributor.occupation'...")
    na_count = original_df['most.recent.contributor.occupation'].isna().sum()
    na_percentage = (na_count / original_rows) * 100
    logger.info(f"NA values: {na_count}")
    logger.info(f"Percentage of NA values: {na_percentage:.2f}%")
    
    # 3. Unique values in 'most.recent.contributor.occupation'
    logger.info("Calculating unique values in 'most.recent.contributor.occupation'...")
    unique_occupations = original_df['most.recent.contributor.occupation'].nunique(dropna=True)
    logger.info(f"Unique values: {unique_occupations}")
    
    # 4. "insufficient_information_gpt" values in the merged dataset
    logger.info("Counting 'insufficient_information_gpt' values...")
    insufficient_count = (merged_df['decoded_classification'] == "insufficient_information_gpt").sum()
    insufficient_percentage = (insufficient_count / merged_rows) * 100
    logger.info(f"Insufficient information count: {insufficient_count}")
    logger.info(f"Percentage of insufficient information: {insufficient_percentage:.2f}%")
    
    # 5. Attempt to reduce "insufficient_information_gpt" values
    updated_count = decode_insufficient_info(merged_df, original_df, decoder_map)
    new_insufficient_count = (merged_df['decoded_classification'] == "insufficient_information_gpt").sum()
    new_insufficient_percentage = (new_insufficient_count / merged_rows) * 100
    
    # Log final statistics
    logger.info(f"Updated 'insufficient_information_gpt' count: {new_insufficient_count}")
    logger.info(f"Percentage of 'insufficient_information_gpt' after decoding: {new_insufficient_percentage:.2f}%")
    
    # Save the updated merged dataset
    updated_path = Path(merged_path).parent / f"{Path(merged_path).stem}_updated.csv"
    merged_df.to_csv(updated_path, index=False)
    logger.info(f"Updated dataset saved to: {updated_path}")
    
    # Print results to the terminal
    print("\n=== Descriptive Statistics ===")
    print(f"1. Original dataset rows: {original_rows}")
    print(f"   Merged dataset rows: {merged_rows}")
    print(f"   Dataset row counts match: {original_rows == merged_rows}")
    print(f"2. NA values in 'most.recent.contributor.occupation': {na_count}")
    print(f"   Percentage of NA values: {na_percentage:.2f}%")
    print(f"3. Unique values in 'most.recent.contributor.occupation': {unique_occupations}")
    print(f"4. Initial insufficient information count: {insufficient_count}")
    print(f"   Initial percentage of insufficient information: {insufficient_percentage:.2f}%")
    print(f"5. Rows successfully decoded: {updated_count}")
    print(f"6. Final insufficient information count: {new_insufficient_count}")
    print(f"   Final percentage of insufficient information: {new_insufficient_percentage:.2f}%")

def main():
    # File paths
    original_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\dime_contributors_2022_individuals.csv"
    merged_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\dime_contributors_2022_individuals_merged.csv"
    decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"

    # Run descriptive statistics and decoding
    descriptive_statistics(original_path, merged_path, decoder_path)

if __name__ == "__main__":
    main()
