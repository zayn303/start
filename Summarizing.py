import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model for both conversation and summarization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

chat_history_ids = None

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        break

    # If user requests a summary of the conversation
    if user_input.lower() == 'summarize':
        if chat_history_ids is not None:
            # Decode the conversation history into text for summarization
            history_text = tokenizer.decode(chat_history_ids, skip_special_tokens=True)
            # Encode the conversation history for summarization
            input_ids = tokenizer.encode("summarize: " + history_text, return_tensors="pt")
            # Generate the summary
            summary_ids = model.generate(input_ids, max_length=100, min_length=20, length_penalty=2.0, num_beams=4)
            # Decode and print the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"Summary: {summary}")
        else:
            print("No conversation history to summarize.")
        continue

    # Tokenize the user input and add it to the conversation history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Update or initialize the conversation history
    if chat_history_ids is None:
        chat_history_ids = new_user_input_ids  # First user input
    else:
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate a response using the same model
    response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and display the response
    response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")
