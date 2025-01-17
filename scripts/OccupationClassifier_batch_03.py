import os
import openai
import json
import pandas as pd
from pathlib import Path
import shutil
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

def get_classification(client, occup_title: str) -> str:
    """Fetch classification from the OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7qnGb8rm",  # Replace with your fine-tuned model ID
            messages=[
                {"role": "system", "content": "classify this entry:"},
                {"role": "user", "content": occup_title}
            ],
            max_tokens=50,
            temperature=0.1
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        logger.error(f"API call failed: {str(e)}")
        return f"Error: {str(e)}"

def process_entry(row: pd.Series, occup_col: str, decoder_map: Dict[str, str]) -> Dict[str, Union[str, None]]:
    """Process a single row and classify the occupation title."""
    def attempt_classification():
        raw_output = get_classification(openai, row[occup_col])
        return raw_output, decoder_map.get(raw_output, None)

    raw_classification, human_readable = attempt_classification()

    if not human_readable:
        raw_classification, human_readable = attempt_classification()

    if not human_readable:
        raw_classification, human_readable = 'insufficient_information_gpt', 'insufficient_information_gpt'

    result = row.to_dict()
    result['raw_classification'] = raw_classification
    result['decoded_classification'] = human_readable
    return result

def process_batch(batch_df: pd.DataFrame, decoder_map: Dict[str, str], occup_col: str) -> List[Dict[str, Union[str, None]]]:
    """Process a batch of rows and classify each one."""
    return [process_entry(row, occup_col, decoder_map) for _, row in batch_df.iterrows()]

def save_batch_results(results: List[Dict[str, Union[str, None]]], temp_file: str, file_format: str) -> None:
    """Save the processed results in the specified format."""
    if file_format == 'jsonl':
        with open(temp_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    else:
        pd.DataFrame(results).to_csv(temp_file, index=False)

def combine_files(temp_dir: Path, output_path: str, file_format: str) -> None:
    """Combine temporary files into a single output file."""
    if file_format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for temp_file in sorted(temp_dir.glob(f'batch_*.{file_format}')):
                with open(temp_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
    else:
        combined_df = pd.concat([pd.read_csv(temp_file) for temp_file in sorted(temp_dir.glob(f'batch_*.{file_format}'))], ignore_index=True)
        combined_df.to_csv(output_path, index=False)

def get_user_input(prompt: str, valid_options: List[str] = None) -> str:
    """Get validated input from the user."""
    while True:
        response = input(prompt).strip()
        if not valid_options or response.lower() in valid_options:
            return response
        logger.warning(f"Invalid input: {response}. Valid options are: {valid_options}")

def select_column(df: pd.DataFrame, prompt: str) -> str:
    """Prompt user to select a column from the DataFrame."""
    logger.info("Available columns:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"{i}. {col}")
    while True:
        try:
            selection = int(input(prompt)) - 1
            if 0 <= selection < len(df.columns):
                return df.columns[selection]
        except ValueError:
            logger.warning("Please enter a valid number.")

def main():
    """Main function for processing occupation classifications."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    format_choice = get_user_input("Choose output format (csv/jsonl): ", valid_options=['csv', 'jsonl'])

    default_decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
    decoder_path = get_user_input(f"Enter the path to your decoder JSONL file (default: {default_decoder_path}): ") or default_decoder_path
    decoder_map = load_decoder(decoder_path)

    input_path = get_user_input("Enter the path to your input CSV file: ")
    df = pd.read_csv(input_path)

    occup_col = select_column(df, "Select the occupation title column number: ")

    input_path_obj = Path(input_path)
    default_output = input_path_obj.parent / f"{input_path_obj.stem}_classified.{format_choice}"

    output_path = get_user_input(f"Enter custom output path (default: {default_output}): ") or default_output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp_classifier"
    temp_dir.mkdir(exist_ok=True)

    batch_size = int(get_user_input("Enter batch size (recommended: 50): "))

    for batch_num in range((len(df) + batch_size - 1) // batch_size):
        batch_df = df.iloc[batch_num * batch_size:(batch_num + 1) * batch_size]
        results = process_batch(batch_df, decoder_map, occup_col)
        temp_file = temp_dir / f"batch_{batch_num}.{format_choice}"
        save_batch_results(results, temp_file, format_choice)

    combine_files(temp_dir, output_path, format_choice)
    shutil.rmtree(temp_dir)

    logger.info(f"Processing complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
