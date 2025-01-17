import os
import json
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_decoder(decoder_path: str) -> dict:
    """Load the decoder map from a JSONL file."""
    decoder_map = {}
    with open(decoder_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            decoder_map[entry["transformed_completion"].strip()] = entry["completion"]
    return decoder_map

def load_all_batches(temp_dir: Path) -> pd.DataFrame:
    """Load all batch files and combine them into a single DataFrame."""
    batch_files = sorted(temp_dir.glob("batch_*.csv"))
    if not batch_files:
        raise ValueError("No batch files found in the temporary directory.")
    logger.info(f"Found {len(batch_files)} batch files.")
    return pd.concat((pd.read_csv(f) for f in batch_files), ignore_index=True)

def decode_classifications(classification_df: pd.DataFrame, decoder_map: dict) -> pd.DataFrame:
    """Decode the raw classifications using the decoder map."""
    classification_df['decoded_classification'] = classification_df['raw_classification'].map(decoder_map).fillna("insufficient_information_gpt")
    return classification_df

def calculate_insufficient_information(classification_df: pd.DataFrame) -> None:
    """Calculate and log the percentage and count of 'insufficient_information_gpt' values."""
    total_count = len(classification_df)
    insufficient_count = (classification_df['decoded_classification'] == "insufficient_information_gpt").sum()
    percentage = (insufficient_count / total_count) * 100
    logger.info(f"Total entries: {total_count}")
    logger.info(f"Insufficient information: {insufficient_count} ({percentage:.2f}%)")

def main():
    """Main function to decode classifications and calculate insufficient information statistics."""
    # Paths
    temp_dir = Path(r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\temp_batches")
    decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
    output_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\contribs_2022\decoded_classifications.csv"

    # Load batch results
    logger.info("Loading batch results...")
    classification_df = load_all_batches(temp_dir)

    # Load decoder
    logger.info("Loading decoder file...")
    decoder_map = load_decoder(decoder_path)

    # Decode classifications
    logger.info("Decoding classifications...")
    classification_df = decode_classifications(classification_df, decoder_map)

    # Calculate and log insufficient information statistics
    logger.info("Calculating insufficient information statistics...")
    calculate_insufficient_information(classification_df)

    # Save the decoded classifications
    logger.info(f"Saving decoded classifications to: {output_path}")
    classification_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
