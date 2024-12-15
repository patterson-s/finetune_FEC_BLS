import os
import openai
import json
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

def load_decoder(decoder_path):
    decoder_map = {}
    with open(decoder_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            decoder_map[entry["transformed_completion"].strip()] = entry["completion"]
    return decoder_map

def get_classification(client, occup_title):
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
        raw_output = response['choices'][0]['message']['content'].strip()
        return raw_output
    except openai.OpenAIError as e:
        return f"Error: {str(e)}"

def process_entry(occup_title, decoder_map):
    raw_classification = get_classification(openai, occup_title)
    human_readable = decoder_map.get(raw_classification, None)
    return {
        'occupation': occup_title,
        'raw_classification': raw_classification,
        'decoded_classification': human_readable if human_readable else "No match found"
    }

def process_batch(batch_df, decoder_map, occup_col):
    results = []
    for _, row in batch_df.iterrows():
        result = process_entry(row[occup_col], decoder_map)
        results.append(result)
    return results

def main():
    # Load OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    # Get decoder file path with default option
    default_decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
    while True:
        decoder_input = input(f"\nEnter the path to your decoder JSONL file (or press '0' to use default path: {default_decoder_path}): ").strip()
        
        if decoder_input == '0':
            decoder_path = default_decoder_path
            break
        elif os.path.exists(decoder_input):
            decoder_path = decoder_input
            break
        print("File not found. Please try again or press '0' for default path.")

    # Load decoder map
    print("Loading decoder map...")
    decoder_map = load_decoder(decoder_path)
    print(f"Loaded {len(decoder_map)} decoder entries")

    # Get input file path
    while True:
        input_path = input("\nEnter the path to your input CSV file: ").strip()
        if os.path.exists(input_path):
            break
        print("File not found. Please try again.")

    # Load dataset and show columns
    df = pd.read_csv(input_path)
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

    # Get column selection for occupation titles
    def get_column_selection(prompt):
        while True:
            try:
                selection = int(input(prompt)) - 1
                if 0 <= selection < len(df.columns):
                    return df.columns[selection]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

    occup_col = get_column_selection("\nSelect the occupation title column number: ")

    # Set up output path
    input_path_obj = Path(input_path)
    default_output = input_path_obj.parent / f"{input_path_obj.stem}_classified.jsonl"
    print(f"\nDefault output path: {default_output}")
    if input("Use this path? (y/n): ").lower() != 'y':
        output_path = input("Enter custom output path: ")
    else:
        output_path = default_output

    # Create temp directory
    temp_dir = Path(output_path).parent / "temp_classifier"
    temp_dir.mkdir(exist_ok=True)

    # Get batch size
    while True:
        try:
            batch_size = int(input("\nEnter batch size (recommended: 50): "))
            if batch_size > 0:
                break
            print("Batch size must be positive.")
        except ValueError:
            print("Please enter a valid number.")

    # Show example outputs
    print("\nProcessing 3 example entries...")
    example_df = df.head(3)
    for _, row in example_df.iterrows():
        result = process_entry(row[occup_col], decoder_map)
        print("\nInput:", result['occupation'])
        print("Raw Classification:", result['raw_classification'])
        print("Decoded Classification:", result['decoded_classification'])
        print("-" * 80)

    if input("\nContinue with this output format? (y/n): ").lower() != 'y':
        print("Exiting...")
        return

    # Process in batches
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    print(f"\nProcessing {len(df)} entries in {total_batches} batches...")

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        
        print(f"\nProcessing batch {batch_num + 1}/{total_batches}")
        batch_df = df.iloc[start_idx:end_idx]
        
        results = process_batch(batch_df, decoder_map, occup_col)
        
        # Save batch results
        temp_file = temp_dir / f"batch_{batch_num}.jsonl"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Saved batch {batch_num + 1} results to {temp_file}")

    # Combine all temp files
    print("\nCombining temporary files...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for temp_file in sorted(temp_dir.glob('batch_*.jsonl')):
            with open(temp_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print(f"\nProcessing complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()