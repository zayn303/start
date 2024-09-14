import os
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the .env file
hf_token = os.getenv("huggingface_token")

if not hf_token:
    raise ValueError("Hugging Face token not found. Make sure it's set in the .env file.")

# Log in using the token
login(hf_token)

# Set up Hugging Face Hub details
repo_id = "zayn303/tuned-T5-small"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load dataset with streaming mode enabled
dataset = load_dataset("jordiclive/wikipedia-summary-dataset", streaming=True)

# Preprocessing function to tokenize input (full_text) and target (summary)
def preprocess_function(examples):
    # Input: full_text (the article)
    inputs = ["summarize: " + doc for doc in examples["full_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # Target: summary (the summary of the article)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Split streaming dataset into training and validation
def split_streaming_dataset(dataset, split_ratio=0.9):
    dataset = list(dataset)  # Convert the streamed dataset into a list
    split_idx = int(len(dataset) * split_ratio)
    return dataset[:split_idx], dataset[split_idx:]

# Since there's no validation set in the dataset, we manually split the training data
streamed_dataset = dataset['train']
train_data, validation_data = split_streaming_dataset(streamed_dataset, split_ratio=0.9)

# Preprocess both the train and validation datasets
train_dataset = map(preprocess_function, train_data)
validation_dataset = map(preprocess_function, validation_data)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer and push them to the Hugging Face Hub
model.save_pretrained(repo_id, push_to_hub=True)
tokenizer.save_pretrained(repo_id, push_to_hub=True)
