import os
import openai
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('decoder_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_decoder(decoder_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    decoder_map = {}
    prompt_map = {}
    try:
        with open(decoder_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                decoder_map[entry["transformed_completion"].strip()] = entry["completion"]
                prompt_occupation = entry["prompt_occupation"].lower().strip()
                if prompt_occupation not in prompt_map:
                    prompt_map[prompt_occupation] = entry["transformed_completion"]
        logger.info(f"Loaded {len(decoder_map)} decoder mappings and {len(prompt_map)} prompt mappings")
        return decoder_map, prompt_map
    except Exception as e:
        logger.error(f"Error loading decoder file: {str(e)}")
        raise

def retry_classification(client, occupation: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
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
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Max retries reached for occupation '{occupation}': {str(e)}")
                return "Error"
            logger.warning(f"Retry attempt {attempt + 1} failed for '{occupation}': {str(e)}")
            time.sleep(2 ** attempt)
    return "Error"

def batch_decode_with_fallback(
    batch_df: pd.DataFrame,
    decoder_map: Dict[str, str],
    prompt_map: Dict[str, str],
    client
) -> pd.DataFrame:
    # Initialize all columns
    batch_df['initial_completion'] = pd.NA
    batch_df['prompt_match_completion'] = pd.NA
    batch_df['retry_transformed_completion'] = pd.NA
    batch_df['retry_completion'] = pd.NA
    batch_df['final_completion'] = pd.NA
    batch_df['decode_note'] = pd.NA
    
    # Step 1: Try standard decoding
    batch_df['initial_completion'] = batch_df['transformed_completion'].map(decoder_map)
    standard_success_mask = batch_df['initial_completion'].notna()
    batch_df.loc[standard_success_mask, 'decode_note'] = 'standard_decode'
    batch_df.loc[standard_success_mask, 'final_completion'] = batch_df.loc[standard_success_mask, 'initial_completion']
    
    # Get failed decodes for next steps
    failed_mask = ~standard_success_mask
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
                batch_df.loc[idx, 'prompt_match_completion'] = completion
                batch_df.loc[idx, 'final_completion'] = completion
                batch_df.loc[idx, 'decode_note'] = 'prompt_match_decode'
    
    # Step 3: Retry classification for remaining failed entries
    still_failed_mask = (batch_df['final_completion'].isna())
    if still_failed_mask.any():
        retry_occupations = batch_df[still_failed_mask]['most.recent.contributor.occupation'].tolist()
        retry_results = []
        for occupation in retry_occupations:
            new_transformed = retry_classification(client, occupation)
            retry_results.append({
                'most.recent.contributor.occupation': occupation,
                'transformed_completion': new_transformed
            })
        
        for result in retry_results:
            occupation = result['most.recent.contributor.occupation']
            new_transformed = result['transformed_completion']
            completion = decoder_map.get(new_transformed)
            
            idx = batch_df[batch_df['most.recent.contributor.occupation'] == occupation].index[0]
            batch_df.loc[idx, 'retry_transformed_completion'] = new_transformed
            
            if completion:
                batch_df.loc[idx, 'retry_completion'] = completion
                batch_df.loc[idx, 'final_completion'] = completion
                batch_df.loc[idx, 'decode_note'] = 'retry_decode'
    
    # Fill any remaining NAs in final_completion with insufficient_information_gpt
    final_failed_mask = batch_df['final_completion'].isna()
    batch_df.loc[final_failed_mask, 'final_completion'] = 'insufficient_information_gpt'
    batch_df.loc[final_failed_mask, 'decode_note'] = 'insufficient_information_gpt'
    
    return batch_df

def get_processed_batches(output_dir: Path) -> set:
    return {int(f.stem.split('_')[1]) for f in output_dir.glob("decoded_batch_*.csv")}

def get_batch_number(filename: str) -> int:
    return int(filename.split('_')[1].split('.')[0])

def main():
    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    # Get paths from user
    default_decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
    decoder_path = input(f"Enter the path to your decoder JSONL file (default: {default_decoder_path}): ").strip() or default_decoder_path
    
    raw_batches_dir = input("Enter the path to your raw batches directory: ").strip()
    raw_batches_dir = Path(raw_batches_dir)
    
    output_dir = input("Enter the path for output directory: ").strip()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load decoder maps
    decoder_map, prompt_map = load_decoder(decoder_path)

    # Get list of batch files and already processed batches
    batch_files = sorted(raw_batches_dir.glob("batch_*.csv"), key=lambda x: get_batch_number(x.name))
    processed_batches = get_processed_batches(output_dir)
    
    remaining_batches = [f for f in batch_files if get_batch_number(f.name) not in processed_batches]
    
    if not remaining_batches:
        logger.info("No new batches to process")
        return

    # Process batches with progress tracking
    total_batches = len(remaining_batches)
    start_time = time.time()
    
    columns_to_track = [
        'most.recent.contributor.occupation',
        'transformed_completion',
        'initial_completion',
        'prompt_match_completion',
        'retry_transformed_completion',
        'retry_completion',
        'final_completion',
        'decode_note'
    ]
    
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for batch_file in remaining_batches:
            try:
                # Load and process batch
                batch_number = get_batch_number(batch_file.name)
                
                # Read and rename columns if necessary
                df = pd.read_csv(batch_file)
                if 'occupation' in df.columns:
                    df = df.rename(columns={
                        'occupation': 'most.recent.contributor.occupation',
                        'raw_classification': 'transformed_completion'
                    })
                
                processed_df = batch_decode_with_fallback(df, decoder_map, prompt_map, openai)
                
                # Ensure all columns are present in the correct order
                for col in columns_to_track:
                    if col not in processed_df.columns:
                        processed_df[col] = pd.NA
                processed_df = processed_df[columns_to_track]
                
                # Save processed batch
                output_file = output_dir / f"decoded_batch_{batch_number}.csv"
                processed_df.to_csv(output_file, index=False)
                
                # Update progress and log statistics
                pbar.update(1)
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / pbar.n
                remaining_time = avg_time_per_batch * (total_batches - pbar.n)
                
                # Calculate and log batch statistics
                batch_total = len(processed_df)
                batch_stats = processed_df['decode_note'].value_counts()
                
                logger.info(f"\nCompleted batch {batch_number}")
                logger.info(f"Estimated time remaining: {remaining_time/60:.2f} minutes")
                logger.info("\nBatch Statistics:")
                logger.info(f"Total records in batch: {batch_total}")
                
                for method in ['standard_decode', 'prompt_match_decode', 'retry_decode', 'insufficient_information_gpt']:
                    count = batch_stats.get(method, 0)
                    percentage = (count/batch_total) * 100
                    logger.info(f"{method}: {count} ({percentage:.2f}%)")
                
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_file}: {str(e)}")
                continue

    # Generate final report
    try:
        all_processed_files = sorted(output_dir.glob("decoded_batch_*.csv"))
        all_results = pd.concat([pd.read_csv(f) for f in all_processed_files])
        
        total_records = len(all_results)
        decode_stats = all_results['decode_note'].value_counts()
        retry_stats = all_results['retry_transformed_completion'].notna().sum()
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Total records processed: {total_records}")
        logger.info("\nDecoding method statistics:")
        for method, count in decode_stats.items():
            percentage = (count/total_records) * 100
            logger.info(f"{method}: {count} ({percentage:.2f}%)")
        logger.info(f"\nRetries attempted: {retry_stats} ({(retry_stats/total_records)*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error generating final report: {str(e)}")

if __name__ == "__main__":
    main()