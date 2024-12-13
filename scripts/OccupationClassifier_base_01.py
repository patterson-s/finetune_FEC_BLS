import os
import openai

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

openai.api_key = api_key

# Classification function using the fine-tuned model
def get_classification(occup_title):
    try:
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7qnGb8rm",
            messages=[
                {"role": "system", "content": "classify this entry:"},
                {"role": "user", "content": occup_title}
            ],
            max_tokens=50,
            temperature=0.1
        )
        return response['choices'][0]['message']['content']
    except openai.OpenAIError as e:
        return f"Error: {str(e)}"

# Main execution loop
if __name__ == "__main__":
    print("Welcome to the occupation classifier powered by your fine-tuned GPT-3.5 model.")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("Enter an occupation: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        classification = get_classification(user_input)
        print(f"Classification: {classification}")
