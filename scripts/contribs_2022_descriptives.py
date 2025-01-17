import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def descriptive_statistics(original_path: str, merged_path: str):
    """Generate descriptive statistics for the original and merged datasets."""
    
    # Load datasets
    logger.info("Loading the original dataset...")
    original_df = pd.read_csv(original_path)
    
    logger.info("Loading the merged dataset...")
    merged_df = pd.read_csv(merged_path)
    
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
    
    # Print results to the terminal
    print("\n=== Descriptive Statistics ===")
    print(f"1. Original dataset rows: {original_rows}")
    print(f"   Merged dataset rows: {merged_rows}")
    print(f"   Dataset row counts match: {original_rows == merged_rows}")
    print(f"2. NA values in 'most.recent.contributor.occupation': {na_count}")
    print(f"   Percentage of NA values: {na_percentage:.2f}%")
    print(f"3. Unique values in 'most.recent.contributor.occupation': {unique_occupations}")
    print(f"4. Insufficient information count: {insufficient_count}")
    print(f"   Percentage of insufficient information: {insufficient_percentage:.2f}%")

def main():
    # File paths
    original_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\dime_contributors_2022_individuals.csv"
    merged_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\dime_contributors_2022_individuals_merged.csv"

    # Run descriptive statistics
    descriptive_statistics(original_path, merged_path)

if __name__ == "__main__":
    main()
