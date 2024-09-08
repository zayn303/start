import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

chat_history = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        break

    # If user requests a summary of the conversation
    if user_input.lower() == 'summarize':
        if chat_history:
            history_text = " ".join(chat_history)
            input_ids = tokenizer.encode("summarize: " + history_text, return_tensors="pt")
            summary_ids = model.generate(input_ids, max_length=100, min_length=20, length_penalty=2.0, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"Summary: {summary}")
        else:
            print("No conversation history to summarize.")
        continue

    # Tokenize the user input and add it to the conversation history
    chat_history.append(user_input)
    new_user_input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Generate a response
    response_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    print(f"Bot: {response}")
    
    # Add the bot's response to the conversation history
    chat_history.append(response)
