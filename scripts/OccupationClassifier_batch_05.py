import os
import openai
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_decoder(decoder_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    decoder_map = {}
    prompt_map = {}
    with open(decoder_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            decoder_map[entry["transformed_completion"].strip()] = entry["completion"]
            # Store first occurrence of each prompt_occupation
            prompt_occupation = entry["prompt_occupation"].lower().strip()
            if prompt_occupation not in prompt_map:
                prompt_map[prompt_occupation] = entry["transformed_completion"]
    return decoder_map, prompt_map

def preprocess_occupation(title: str) -> str:
    if not isinstance(title, str):
        return "unknown"
    return title.strip()

def get_classifications(client, occupations: List[str]) -> List[Dict[str, str]]:
    results = []
    for occupation in occupations:
        try:
            response = client.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:personal::7qnGb8rm",
                messages=[
                    {"role": "system", "content": "classify this entry:"},
                    {"role": "user", "content": occupation}
                ],
                max_tokens=50,
                temperature=0.1
            )
            transformed_completion = response['choices'][0]['message']['content'].strip()
            results.append({
                "most.recent.contributor.occupation": occupation,
                "transformed_completion": transformed_completion
            })
        except openai.OpenAIError as e:
            logger.error(f"API call failed for occupation '{occupation}': {str(e)}")
            results.append({
                "most.recent.contributor.occupation": occupation,
                "transformed_completion": "Error"
            })
    return results

def batch_decode_with_fallback(
    batch_df: pd.DataFrame,
    decoder_map: Dict[str, str],
    prompt_map: Dict[str, str],
    client
) -> pd.DataFrame:
    # Step 1: Try standard decoding
    batch_df['completion'] = batch_df['transformed_completion'].map(decoder_map)
    batch_df['decode_note'] = 'standard_decode'
    
    # Identify failed decodes
    failed_mask = batch_df['completion'].isna()
    if not failed_mask.any():
        return batch_df
    
    # Step 2: Try prompt matching for failed entries
    failed_df = batch_df[failed_mask].copy()
    for idx, row in failed_df.iterrows():
        original_occupation = row['most.recent.contributor.occupation'].lower().strip()
        if original_occupation in prompt_map:
            transformed_completion = prompt_map[original_occupation]
            completion = decoder_map.get(transformed_completion)
            if completion:
                batch_df.loc[idx, 'completion'] = completion
                batch_df.loc[idx, 'decode_note'] = 'prompt_match_decode'
    
    # Step 3: Retry classification for remaining failed entries
    still_failed_mask = batch_df['completion'].isna()
    if still_failed_mask.any():
        retry_occupations = batch_df[still_failed_mask]['most.recent.contributor.occupation'].tolist()
        retry_results = get_classifications(client, retry_occupations)
        
        for result in retry_results:
            occupation = result['most.recent.contributor.occupation']
            new_transformed = result['transformed_completion']
            completion = decoder_map.get(new_transformed)
            
            idx = batch_df[batch_df['most.recent.contributor.occupation'] == occupation].index[0]
            if completion:
                batch_df.loc[idx, 'completion'] = completion
                batch_df.loc[idx, 'decode_note'] = 'retry_decode'
            else:
                batch_df.loc[idx, 'completion'] = 'insufficient_information_gpt'
                batch_df.loc[idx, 'decode_note'] = 'insufficient_information_gpt'
    
    # Set any remaining NaN completions to insufficient_information_gpt
    final_failed_mask = batch_df['completion'].isna()
    batch_df.loc[final_failed_mask, 'completion'] = 'insufficient_information_gpt'
    batch_df.loc[final_failed_mask, 'decode_note'] = 'insufficient_information_gpt'
    
    return batch_df

def save_batch(batch_results: List[Dict[str, str]], batch_number: int, temp_dir: Path) -> None:
    batch_file = temp_dir / f"batch_{batch_number}.csv"
    pd.DataFrame(batch_results).to_csv(batch_file, index=False)
    logger.info(f"Saved batch {batch_number} to {batch_file}")

def load_existing_batches(temp_dir: Path) -> pd.DataFrame:
    batch_files = sorted(temp_dir.glob("batch_*.csv"))
    if not batch_files:
        return pd.DataFrame(columns=["most.recent.contributor.occupation", "transformed_completion"])
    
    try:
        # Try to load existing batches
        existing_df = pd.concat((pd.read_csv(f) for f in batch_files), ignore_index=True)
        
        # If old column names exist, rename them
        column_mapping = {
            'occupation': 'most.recent.contributor.occupation',
            'raw_classification': 'transformed_completion'
        }
        existing_df.rename(columns=column_mapping, inplace=True)
        
        # Ensure required columns exist
        required_columns = ["most.recent.contributor.occupation", "transformed_completion"]
        missing_columns = [col for col in required_columns if col not in existing_df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in existing batches: {missing_columns}")
            logger.warning("Clearing existing batches directory and starting fresh")
            for f in batch_files:
                f.unlink()
            return pd.DataFrame(columns=required_columns)
            
        return existing_df
        
    except Exception as e:
        logger.warning(f"Error loading existing batches: {str(e)}")
        logger.warning("Clearing existing batches directory and starting fresh")
        for f in batch_files:
            f.unlink()
        return pd.DataFrame(columns=["most.recent.contributor.occupation", "transformed_completion"])

def process_unique_occupations_in_batches(
    unique_occupations: List[str],
    batch_size: int,
    temp_dir: Path,
    decoder_map: Dict[str, str],
    prompt_map: Dict[str, str]
) -> pd.DataFrame:
    existing_results = load_existing_batches(temp_dir)
    processed_occupations = set(existing_results['most.recent.contributor.occupation'])
    results = existing_results.to_dict('records')
    
    num_batches = (len(unique_occupations) + batch_size - 1) // batch_size
    
    # TESTING ONLY - Limit to 2 batches
    # Comment out these 2 lines for production
    num_batches = min(num_batches, 2)
    unique_occupations = unique_occupations[:batch_size * 2]
    
    with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, len(unique_occupations), batch_size):
            batch_number = i // batch_size
            batch = [occ for occ in unique_occupations[i:i + batch_size] if occ not in processed_occupations]
            if not batch:
                pbar.update(1)
                continue
                
            # Get initial classifications
            batch_results = get_classifications(openai, batch)
            batch_df = pd.DataFrame(batch_results)
            
            # Apply decoding with fallback
            decoded_df = batch_decode_with_fallback(batch_df, decoder_map, prompt_map, openai)
            
            # Save batch and update results
            save_batch(decoded_df.to_dict('records'), batch_number, temp_dir)
            results.extend(decoded_df.to_dict('records'))
            pbar.update(1)
            
    return pd.DataFrame(results)

def get_user_input(prompt: str, valid_options: List[str] = None) -> str:
    while True:
        response = input(prompt).strip()
        if not valid_options or response.lower() in valid_options:
            return response
        logger.warning(f"Invalid input: {response}. Valid options are: {valid_options}")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    format_choice = "csv"
    default_decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
    decoder_path = get_user_input(f"Enter the path to your decoder JSONL file (default: {default_decoder_path}): ") or default_decoder_path
    decoder_map, prompt_map = load_decoder(decoder_path)

    input_path = get_user_input("Enter the path to your input CSV file: ")
    df = pd.read_csv(input_path)

    if 'most.recent.contributor.occupation' not in df.columns:
        raise ValueError("'most.recent.contributor.occupation' column not found in the input file.")

    if df['most.recent.contributor.occupation'].isnull().any():
        df = df.dropna(subset=['most.recent.contributor.occupation'])

    unique_occupations = df['most.recent.contributor.occupation'].drop_duplicates().tolist()

    temp_dir = Path(input_path).parent / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 100
    classification_results = process_unique_occupations_in_batches(
        unique_occupations,
        batch_size,
        temp_dir,
        decoder_map,
        prompt_map
    )

    crosswalk = classification_results.set_index('most.recent.contributor.occupation')
    df['transformed_completion'] = df['most.recent.contributor.occupation'].map(crosswalk['transformed_completion'])
    df['completion'] = df['most.recent.contributor.occupation'].map(crosswalk['completion'])
    df['decode_note'] = df['most.recent.contributor.occupation'].map(crosswalk['decode_note'])

    # Get output directory from user
    output_dir = get_user_input("Enter the path to your output directory: ")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get output filename from user, with a default suggestion
    input_path_obj = Path(input_path)
    default_filename = f"{input_path_obj.stem}_classified.{format_choice}"
    suggested_output = output_dir / default_filename
    
    output_path = get_user_input(f"Enter the output filename (default: {default_filename}): ") or default_filename
    final_output_path = output_dir / output_path
    
    # Check if file exists
    if final_output_path.exists():
        overwrite = get_user_input(f"File {final_output_path} already exists. Overwrite? (y/n): ", ['y', 'n'])
        if overwrite.lower() != 'y':
            logger.info("Operation cancelled by user")
            return
    
    df.to_csv(final_output_path, index=False)

    # Log summary statistics
    total_records = len(df)
    decode_stats = df['decode_note'].value_counts()
    logger.info("\nProcessing complete! Summary statistics:")
    logger.info(f"Total records processed: {total_records}")
    for method, count in decode_stats.items():
        percentage = (count/total_records) * 100
        logger.info(f"{method}: {count} ({percentage:.2f}%)")
    logger.info(f"\nResults saved to: {default_output}")

if __name__ == "__main__":
    main()