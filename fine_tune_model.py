import os
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login, Repository

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

# Load and preprocess the dataset with streaming enabled
dataset = load_dataset("jordiclive/wikipedia-summary-dataset", streaming=True)

def preprocess_function(examples):
    # Use 'full_text' as the article (input) and 'summary' as the target (output)
    inputs = ["summarize: " + doc for doc in examples["full_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # Setup the tokenizer for targets (using 'summary')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Stream the dataset and preprocess it
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
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
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the model and push it to Hugging Face Hub
model.save_pretrained(repo_id, push_to_hub=True)
tokenizer.save_pretrained(repo_id, push_to_hub=True)
