import sys, os
import json
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    Trainer, TrainingArguments, DataCollatorWithPadding,
    AutoTokenizer, AutoModelForSequenceClassification, set_seed
)
from src.preprocess import preprocess_texts

# Add project root (one level up) to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA_PATH = "data/raw/social_sentiment.csv"
OUTPUT_DIR = "models/roberta_sentiment_model"

label_map = {"negative": 0, "neutral": 1, "positive": 2}

# initialization
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
collator = DataCollatorWithPadding(tokenizer=tokenizer)


# fucntions
def load_data(path: str):
    """Load and preprocess the dataset."""
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
    """Tokenize text data for model input."""
    dataset = Dataset.from_pandas(df)
    tokenized_ds = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding=True),
        batched=True
    )
    return tokenized_ds


def compute_metrics(eval_pred):
    """Compute accuracy metric during evaluation."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def train_model(tokenized_ds):
    """Train the sentiment model and save metrics and weights."""
    print("Starting training...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )
    
    # Use a small subset of the dataset for quick evaluation
    eval_subset = tokenized_ds.select(range(min(100, len(tokenized_ds))))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        eval_dataset=eval_subset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    # Evaluate model accuracy on eval subset
    metrics = trainer.evaluate()
    acc = metrics.get("eval_accuracy", 0.0)
    print(f"Model accuracy: {acc:.3f}")

    # Save model, tokenizer, and metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f)
    
    print(f"Training completed. Model and metrics saved to {OUTPUT_DIR}")


def main():
    """Main execution entrypoint."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    tokenized_ds = tokenize_data(df)
    train_model(tokenized_ds)


if __name__ == "__main__":
    main()
