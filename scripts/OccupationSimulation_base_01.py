import os
import openai

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

openai.api_key = api_key

# Function to generate a completion for a given occupation and number with streaming
def generate_simulation(occupation, number):
    try:
        # Create the prompt
        prompt = (
            f"Please complete the analogy and include misspellings.\n"
            f"Lawyer (13) = Attorney; Atorney; Attorney-at-Law; Lawer; Patent Lawyer; Corporate Lawyer; "
            f"Finance Lawyer; Human Rights Attorney; lawyer; lawyr; attorne; attorney at law; trial lawyer.\n"
            f"{occupation} ({number}) ="
        )
        
        # Use OpenAI API to generate a response with streaming enabled
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
            stream=True  # Enable streaming
        )
        
        # Stream the response as it arrives
        print("Generated Completion (streaming): ", end="", flush=True)
        completion = ""
        for chunk in response:
            chunk_content = chunk["choices"][0].get("delta", {}).get("content", "")
            completion += chunk_content
            print(chunk_content, end="", flush=True)
        print()  # Add a newline after streaming is complete
        return completion.strip()
    except openai.OpenAIError as e:
        return f"Error: {str(e)}"

# Main script logic
if __name__ == "__main__":
    print("Welcome to the Occupation Simulation Demo!")

    while True:
        # Get occupation input
        occupation = input("Enter an occupation: ").strip()
        if occupation.lower() == "exit":
            print("Goodbye!")
            break

        # Get number input
        try:
            number = int(input("How many entries would you like to simulate?: ").strip())
        except ValueError:
            print("Invalid input. Please provide a valid number.")
            continue

        # Generate simulation
        simulation_result = generate_simulation(occupation, number)

        # Ask if the user wants to try again
        retry = input("Would you like to try again? (yes or no): ").strip().lower()
        if retry != "yes":
            print("Goodbye!")
            break
