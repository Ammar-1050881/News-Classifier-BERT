import os
import numpy as np
import evaluate
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    BertTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# 1. SETUP & AUTHENTICATION
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("⏳ Step 1: Loading AG News Dataset...")
try:
    # Using namespaced version to bypass Windows URI errors
    dataset = load_dataset("fancyzhx/ag_news", token=hf_token)
    print("✅ Dataset Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# 2. TOKENIZATION (The Translator)
print("⏳ Step 2: Preparing BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=64
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. SELECTING SUBSET (For Speed & Deadline)
# We use 2000 for training and 500 for testing to ensure it runs quickly today
train_subset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
test_subset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# 4. MODEL SETUP
print("⏳ Step 3: Initializing BERT Model (4 Categories)...")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=4
)

# 5. METRICS (The Report Card)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./results",          
    eval_strategy="epoch",           # Modern keyword for latest library
    save_strategy="epoch",           
    learning_rate=2e-5,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    num_train_epochs=1,              # 1 Epoch is enough for this test run
    weight_decay=0.01,
    logging_dir='./logs',            
    load_best_model_at_end=True,     
)

# 7. THE TRAINER (Autopilot)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=test_subset,
    compute_metrics=compute_metrics,
)

# 8. START TRAINING
print("\n🚀 Starting Fine-Tuning... (This may take a few minutes)")
trainer.train()

# 9. SAVE THE SMART MODEL
print("\n💾 Saving your custom BERT model...")
model.save_pretrained("./news_classifier_bert")
tokenizer.save_pretrained("./news_classifier_bert")
print("✅ Model saved to folder: './news_classifier_bert'")