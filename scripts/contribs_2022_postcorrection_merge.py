import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def combine_batch_files(batch_dir: Path, output_file: Path) -> None:
    """
    Combine all batch files into a single consolidated file
    """
    logger.info("Starting batch file consolidation...")
    
    # Get list of all batch files
    batch_files = sorted(batch_dir.glob("decoded_batch_*.csv"))
    if not batch_files:
        raise ValueError(f"No batch files found in {batch_dir}")
    
    logger.info(f"Found {len(batch_files)} batch files")
    
    # Read and concatenate all batch files
    dfs = []
    for file in batch_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.info(f"Read {file.name}")
        except Exception as e:
            logger.error(f"Error reading {file.name}: {str(e)}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Save consolidated file
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved consolidated file to {output_file}")

def merge_with_original(
    original_file: Path,
    decoded_file: Path,
    output_file: Path,
    columns_to_merge: list
) -> None:
    """
    Merge decoded data with original dataset
    """
    logger.info("Starting merge with original dataset...")
    
    # Read files
    try:
        original_df = pd.read_csv(original_file)
        logger.info(f"Read original file. Shape: {original_df.shape}")
        
        decoded_df = pd.read_csv(decoded_file)
        logger.info(f"Read decoded file. Shape: {decoded_df.shape}")
    except Exception as e:
        logger.error(f"Error reading files: {str(e)}")
        raise
    
    # Merge dataframes
    try:
        merged_df = original_df.merge(
            decoded_df[['most.recent.contributor.occupation'] + columns_to_merge],
            on='most.recent.contributor.occupation',
            how='left'
        )
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        
        # Check for any rows that didn't get matches
        unmatched = merged_df[columns_to_merge[0]].isna().sum()
        if unmatched > 0:
            logger.warning(f"Found {unmatched} rows without matches")
    except Exception as e:
        logger.error(f"Error during merge: {str(e)}")
        raise
    
    # Save final output
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved complete dataset to {output_file}")

def main():
    # Define paths
    base_dir = Path(r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022")
    batch_dir = base_dir / "contribs_corrected"  # Directory containing the batch files
    
    # Define file paths
    original_file = base_dir / "dime_contributors_2022_individuals.csv"
    consolidated_decode_file = base_dir / "contribs_2022_decode_01.csv"
    final_output_file = base_dir / "contribs_2022_complete_01.csv"
    
    # Columns to merge from decoded data
    columns_to_merge = ['transformed_completion', 'final_completion', 'decode_note']
    
    try:
        # Step 1: Combine batch files
        logger.info("Step 1: Combining batch files")
        combine_batch_files(batch_dir, consolidated_decode_file)
        
        # Step 2: Merge with original dataset
        logger.info("Step 2: Merging with original dataset")
        merge_with_original(
            original_file,
            consolidated_decode_file,
            final_output_file,
            columns_to_merge
        )
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()