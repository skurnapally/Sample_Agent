import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
device = torch.device("cpu")  # Force CPU usage
import pandas as pd

# Step 1: Load and Tokenize the Dataset
def load_and_tokenize_dataset(dataset_path, tokenizer):
    # Load CSV into Pandas DataFrame
    df = pd.read_csv(dataset_path)

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["review"], padding="max_length", truncation=True), batched=True)

    return tokenized_dataset

# Set dataset path and model checkpoint
dataset_path = "reviews.csv"  # Make sure this file exists
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load and tokenize dataset
tokenized_dataset = load_and_tokenize_dataset(dataset_path, tokenizer)

# Step 2: Split the dataset into Train and Test
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Step 3: Define Model and Training Arguments
#model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)  # 3 classes
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3).to(device)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Step 4: Define Evaluation Metrics
# Load accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

# Step 5: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Step 6: Evaluate the Model
trainer.evaluate()

# Step 7: Make Predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_class

# Example usage
print(f"Predicted class: {predict('I love this product!')}")

# Step 8: Save and Load the Model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_tokenizer")

# To load later:
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
# tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
