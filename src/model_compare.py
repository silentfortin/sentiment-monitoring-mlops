import os
import json
from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import evaluate

# Compare a new local sentiment model against the current production model on Hugging Face

repo_id = "CapDimble/sentiment-monitoring-model"  # current production model on HF Hub
new_model_dir = "models/roberta_sentiment_model"  # newly trained local model
new_metrics_path = os.path.join(new_model_dir, "metrics.json")

# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Load evaluation dataset
dataset = load_dataset("tweet_eval", "sentiment", split="test[:200]")
metric = evaluate.load("accuracy")

# Evaluate production model from Hugging Face Hub
print("Evaluating current production model from Hugging Face...")
prod_clf = pipeline("sentiment-analysis", model=repo_id)
prod_preds = [prod_clf(text)[0]["label"] for text in dataset["text"]]
label_map = {"negative": 0, "neutral": 1, "positive": 2}
prod_encoded = [label_map[p] for p in prod_preds]
prod_acc = metric.compute(references=dataset["label"], predictions=prod_encoded)["accuracy"]
print(f"Production model accuracy: {prod_acc:.3f}")

# Evaluate new local model
print("Evaluating new trained model...")
new_clf = pipeline("sentiment-analysis", model=new_model_dir)
new_preds = [new_clf(text)[0]["label"] for text in dataset["text"]]
new_encoded = [label_map[p] for p in new_preds]
new_acc = metric.compute(references=dataset["label"], predictions=new_encoded)["accuracy"]
print(f"New model accuracy: {new_acc:.3f}")

# Save new metrics
os.makedirs(new_model_dir, exist_ok=True)
with open(new_metrics_path, "w") as f:
    json.dump({"accuracy": new_acc}, f, indent=4)

# Compare and decide
if new_acc > prod_acc:
    print("New model performs better. Promoting to production on Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(new_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(new_model_dir)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print("Model successfully updated on Hugging Face.")
else:
    print("New model did not outperform the production model. Keeping current version.")
