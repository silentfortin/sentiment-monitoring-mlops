import sys, os
import pandas as pd
from datasets import Dataset
from transformers import (
    Trainer, TrainingArguments, DataCollatorWithPadding,
    AutoTokenizer, AutoModelForSequenceClassification, set_seed
)
from src.preprocess import preprocess_texts

# === CONFIG ===
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA_PATH = "data/raw/social_sentiment.csv"
OUTPUT_DIR = "models/roberta_sentiment_model"

label_map = {"negative": 0, "neutral": 1, "positive": 2}

# === INITIALIZATION ===
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
collator = DataCollatorWithPadding(tokenizer=tokenizer)


# === FUNCTIONS ===
def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    print(f"Loading dataset from: {path}")
    
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "sentiment"])
    
    print("Preprocessing texts...")
    df["text"] = preprocess_texts(df["text"])
    df["label"] = [label_map[lab] for lab in df["sentiment"]]
    return df


def tokenize_data(df):
    dataset = Dataset.from_pandas(df)
    tokenized_ds = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding=True),
        batched=True
    )
    return tokenized_ds


def train_model(tokenized_ds):
    print("Starting training...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=collator
    )
    
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training completed. Model saved in {OUTPUT_DIR}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    tokenized_ds = tokenize_data(df)
    train_model(tokenized_ds)


if __name__ == "__main__":
    main()