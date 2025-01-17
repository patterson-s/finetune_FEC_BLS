import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_decoded_classifications(original_path: str, decoded_path: str, output_path: str) -> None:
    """Merge decoded classifications back into the original dataset."""
    # Load the original dataset
    logger.info("Loading the original dataset...")
    original_df = pd.read_csv(original_path)

    # Load the decoded classifications
    logger.info("Loading the decoded classifications dataset...")
    decoded_df = pd.read_csv(decoded_path)

    # Perform the merge
    logger.info("Merging decoded classifications into the original dataset...")
    merged_df = original_df.merge(
        decoded_df,
        left_on="most.recent.contributor.occupation",
        right_on="occupation",
        how="left"  # Keep all original rows, even if no match
    )

    # Handle missing classifications
    logger.info("Handling missing classifications...")
    merged_df['raw_classification'] = merged_df['raw_classification'].fillna("insufficient_information_gpt")
    merged_df['decoded_classification'] = merged_df['decoded_classification'].fillna("insufficient_information_gpt")

    # Drop redundant columns (e.g., `occupation` from the decoded dataset)
    merged_df.drop(columns=["occupation"], inplace=True)

    # Save the merged dataset
    logger.info(f"Saving the merged dataset to: {output_path}")
    merged_df.to_csv(output_path, index=False)

def main():
    """Main function to merge classifications back into the original dataset."""
    # File paths
    original_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\dime_contributors_2022_individuals.csv"
    decoded_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\decoded_classifications.csv"
    output_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\dime_contributors_2022_individuals_merged.csv"

    # Merge the datasets
    merge_decoded_classifications(original_path, decoded_path, output_path)

if __name__ == "__main__":
    main()
