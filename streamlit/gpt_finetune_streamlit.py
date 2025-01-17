import os
import openai
import json
import streamlit as st

# ---- PATHS ----
DECODER_PATH = r"C:\Users\spatt\Desktop\finetune_FEC_BLS\data\finetune.jsonl"

# ---- API Key Setup ----
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    st.stop()

openai.api_key = api_key

# ---- Load Decoder File ----
decoder_map = {}
try:
    with open(DECODER_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            decoder_map[entry["transformed_completion"].strip()] = entry["completion"]
except FileNotFoundError:
    st.error(f"Decoder file not found at: {DECODER_PATH}")
    st.stop()

# ---- Classification Function ----
def get_classification(occup_title):
    """Call the fine-tuned GPT-3.5 model and return the raw classification."""
    try:
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7qnGb8rm",  # Replace with your model ID
            messages=[
                {"role": "system", "content": "classify this entry:"},
                {"role": "user", "content": occup_title}
            ],
            max_tokens=50,
            temperature=0.1
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        return f"Error: {str(e)}"

# ---- Decoding Function ----
def decode_classification(raw_output):
    """Decode the raw classification using the decoder map."""
    return decoder_map.get(raw_output, None)  # Return None if no match is found

# ---- Streamlit App ----
def main():
    st.title("Occupation Classifier")
    st.markdown("""
    **Fine-Tuned GPT-3.5 Model**  
    Enter an occupation title, and the model will classify it.  
    - **Raw Classification**: The direct output from the model.  
    - **Human-Readable Classification**: The decoded version using the provided decoder file.  
    """)

    user_input = st.text_input("Enter an occupation title:", "")
    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter a valid occupation title.")
        else:
            # Get classification and decode
            with st.spinner("Classifying..."):
                raw_classification = get_classification(user_input)
                human_readable = decode_classification(raw_classification)

            # Display results
            st.subheader("Results")
            st.write(f"**Raw Classification**: {raw_classification}")
            if human_readable:
                st.write(f"**Human-Readable Classification**: {human_readable}")
            else:
                st.warning(f"No match found for the raw output. Likely hallucination.")
                st.write(f"**Raw Output**: {raw_classification}")

                retry = st.radio("Would you like to try again?", ["Yes", "No"])
                if retry == "No":
                    st.stop()

if __name__ == "__main__":
    main()
