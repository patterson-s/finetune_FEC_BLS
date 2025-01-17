import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """
    Load data and remove duplicates
    """
    logger.info("Loading dataset...")
    df = pd.read_csv(file_path)
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    rows_after_dedup = len(df)
    
    logger.info(f"Removed {initial_rows - rows_after_dedup} duplicate rows")
    return df

def analyze_occupation_nulls(df: pd.DataFrame) -> None:
    """
    Analyze rows with missing occupation values
    """
    # Check nulls in occupation
    null_occupations = df['most.recent.contributor.occupation'].isna()
    null_count = null_occupations.sum()
    
    # Check if these are the same rows that didn't get matches
    unmatched_completions = df['final_completion'].isna()
    unmatched_count = unmatched_completions.sum()
    
    # Check overlap
    overlap_count = (null_occupations & unmatched_completions).sum()
    
    logger.info("\nNull Analysis:")
    logger.info(f"Rows with null occupations: {null_count}")
    logger.info(f"Rows with unmatched completions: {unmatched_count}")
    logger.info(f"Overlap (rows with both null): {overlap_count}")

def generate_descriptives(df: pd.DataFrame) -> None:
    """
    Generate descriptive statistics
    """
    logger.info("\nDescriptive Statistics:")
    
    # 1. Total rows
    total_rows = len(df)
    logger.info(f"\n1. Total number of rows: {total_rows:,}")
    
    # 2. Unique occupations
    unique_occupations = df['most.recent.contributor.occupation'].nunique()
    logger.info(f"\n2. Number of unique occupations: {unique_occupations:,}")
    
    # 3. NA values in occupation
    na_occupations = df['most.recent.contributor.occupation'].isna().sum()
    na_percent = (na_occupations / total_rows) * 100
    logger.info(f"\n3. Number of NA values in occupation: {na_occupations:,} ({na_percent:.2f}%)")
    
    # 4. Insufficient information counts
    insufficient_count = df['final_completion'].eq('insufficient_information_gpt').sum()
    insufficient_percent = (insufficient_count / total_rows) * 100
    logger.info(f"\n4. Number of 'insufficient_information_gpt' values: {insufficient_count:,} ({insufficient_percent:.2f}%)")
    
    # 5. Decode note breakdown
    logger.info("\n5. Decode note breakdown:")
    decode_counts = df['decode_note'].value_counts()
    decode_percentages = (decode_counts / total_rows * 100).round(2)
    
    for note, count in decode_counts.items():
        percentage = decode_percentages[note]
        logger.info(f"   {note}: {count:,} ({percentage:.2f}%)")

def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the cleaned dataset
    """
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved cleaned dataset to {output_path}")

def main():
    # Define file paths
    base_dir = Path(r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022")
    input_file = base_dir / "contribs_2022_complete_01.csv"
    output_file = base_dir / "contribs_2022_complete_01_cleaned.csv"
    
    try:
        # Load and clean data
        df = load_and_clean_data(input_file)
        
        # Run analyses
        analyze_occupation_nulls(df)
        generate_descriptives(df)
        
        # Save cleaned dataset
        save_cleaned_data(df, output_file)
        
        logger.info("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()