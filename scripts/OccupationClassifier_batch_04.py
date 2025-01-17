import os
import openai
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_decoder(decoder_path: str) -> Dict[str, str]:
    """Load the decoder map from a JSONL file."""
    decoder_map = {}
    with open(decoder_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            decoder_map[entry["transformed_completion"].strip()] = entry["completion"]
    return decoder_map

def preprocess_occupation(title: str) -> str:
    """Preprocess occupation title to ensure valid API input."""
    if not isinstance(title, str):
        return "unknown"
    return title.strip()

def get_classifications(client, occupations: List[str]) -> List[Dict[str, str]]:
    """Fetch classifications for a batch of occupations from the OpenAI API."""
    results = []
    for occupation in occupations:
        try:
            response = client.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:personal::7qnGb8rm",  # Replace with your model
                messages=[
                    {"role": "system", "content": "classify this entry:"},
                    {"role": "user", "content": occupation}
                ],
                max_tokens=50,
                temperature=0.1
            )
            raw_classification = response['choices'][0]['message']['content'].strip()
            results.append({"occupation": occupation, "raw_classification": raw_classification})
        except openai.OpenAIError as e:
            logger.error(f"API call failed for occupation '{occupation}': {str(e)}")
            results.append({"occupation": occupation, "raw_classification": "Error"})
    return results

def save_batch(batch_results: List[Dict[str, str]], batch_number: int, temp_dir: Path) -> None:
    """Save the results of a single batch to a temporary file."""
    batch_file = temp_dir / f"batch_{batch_number}.csv"
    pd.DataFrame(batch_results).to_csv(batch_file, index=False)
    logger.info(f"Saved batch {batch_number} to {batch_file}")

def load_existing_batches(temp_dir: Path) -> pd.DataFrame:
    """Load all previously saved batch results."""
    batch_files = sorted(temp_dir.glob("batch_*.csv"))
    if not batch_files:
        return pd.DataFrame(columns=["occupation", "raw_classification"])
    return pd.concat((pd.read_csv(f) for f in batch_files), ignore_index=True)

def process_unique_occupations_in_batches(unique_occupations: List[str], batch_size: int, temp_dir: Path) -> pd.DataFrame:
    """Process unique occupations in batches and save progress incrementally."""
    existing_results = load_existing_batches(temp_dir)
    processed_occupations = set(existing_results['occupation'])
    results = existing_results.to_dict('records')
    
    num_batches = (len(unique_occupations) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, len(unique_occupations), batch_size):
            batch_number = i // batch_size
            batch = [occ for occ in unique_occupations[i:i + batch_size] if occ not in processed_occupations]
            if not batch:
                pbar.update(1)  # Skip completed batch
                continue
            batch_results = get_classifications(openai, batch)
            results.extend(batch_results)
            save_batch(batch_results, batch_number, temp_dir)
            pbar.update(1)
    return pd.DataFrame(results)

def get_user_input(prompt: str, valid_options: List[str] = None) -> str:
    """Get validated input from the user."""
    while True:
        response = input(prompt).strip()
        if not valid_options or response.lower() in valid_options:
            return response
        logger.warning(f"Invalid input: {response}. Valid options are: {valid_options}")

def main():
    """Main function for processing occupation classifications."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    format_choice = "csv"  # Defaulting to CSV for simplicity
    default_decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
    decoder_path = get_user_input(f"Enter the path to your decoder JSONL file (default: {default_decoder_path}): ") or default_decoder_path
    decoder_map = load_decoder(decoder_path)

    input_path = get_user_input("Enter the path to your input CSV file: ")
    df = pd.read_csv(input_path)

    # Validate and preprocess the dataset
    if 'most.recent.contributor.occupation' not in df.columns:
        raise ValueError("'most.recent.contributor.occupation' column not found in the input file.")

    if df['most.recent.contributor.occupation'].isnull().any():
        df = df.dropna(subset=['most.recent.contributor.occupation'])

    unique_occupations = df['most.recent.contributor.occupation'].drop_duplicates().tolist()

    # Temporary directory to save batch progress
    temp_dir = Path(input_path).parent / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process unique occupations in batches
    batch_size = 100  # Adjust batch size based on your API limits and dataset size
    classification_results = process_unique_occupations_in_batches(unique_occupations, batch_size, temp_dir)

    # Add decoder mapping (if needed)
    classification_results['decoded_classification'] = classification_results['raw_classification'].map(decoder_map.get)

    # Create a crosswalk and map back to the original DataFrame
    crosswalk = classification_results.set_index('occupation')
    df['raw_classification'] = df['most.recent.contributor.occupation'].map(crosswalk['raw_classification'])
    df['decoded_classification'] = df['most.recent.contributor.occupation'].map(crosswalk['decoded_classification'])

    # Save the final results
    input_path_obj = Path(input_path)
    default_output = input_path_obj.parent / f"{input_path_obj.stem}_classified.{format_choice}"
    df.to_csv(default_output, index=False)

    logger.info(f"Processing complete! Results saved to: {default_output}")

if __name__ == "__main__":
    main()
