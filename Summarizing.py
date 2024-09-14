from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the .env file
hf_token = os.getenv("huggingface_token")

if not hf_token:
    raise ValueError("Hugging Face token not found. Make sure it's set in the .env file.")

# Log in using the token
login(hf_token)

# Load tokenizer and model for summarization
tokenizer = T5Tokenizer.from_pretrained("tuned-T5-small")
model = T5ForConditionalGeneration.from_pretrained("tuned-T5-small")

# Get text input from the user to summarize
text_to_summarize = input("Please enter the text you want to summarize:\n")

# Preprocess and encode the text for summarization
input_ids = tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt")

# Generate summary
summary_ids = model.generate(input_ids, max_length=60, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"\nSummary: {summary}")