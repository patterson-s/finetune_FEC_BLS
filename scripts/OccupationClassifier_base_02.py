import os
import openai
import json

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

openai.api_key = api_key

# Load the decoder file
decoder_path = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"
decoder_map = {}

# Read the decoder file and build a lookup dictionary
with open(decoder_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        decoder_map[entry["transformed_completion"].strip()] = entry["completion"]

# Classification function using the fine-tuned model
def get_classification(occup_title):
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

# Decoding function
def decode_classification(raw_output):
    return decoder_map.get(raw_output, None)  # Return None if no match is found

# Main execution loop
if __name__ == "__main__":
    print("Welcome to the occupation classifier powered by your fine-tuned GPT-3.5 model.")
    print("Type 'exit' to quit.")
    previous_input = None

    while True:
        if previous_input is None:
            user_input = input("Enter an occupation: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            previous_input = user_input
        else:
            user_input = previous_input

        raw_classification = get_classification(user_input)
        human_readable = decode_classification(raw_classification)

        if human_readable:
            print(f"Raw Classification: {raw_classification}")
            print(f"Human-Readable Classification: {human_readable}")
            previous_input = None  # Clear previous input for the next loop
        else:
            print(f"No match, hallucination likely.\nRaw output: {raw_classification}")
            retry = input("Try again? Write yes or no: ").strip().lower()
            if retry != "yes":
                print("Goodbye!")
                break
