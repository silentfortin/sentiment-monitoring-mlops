import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline

# Configuration
MODEL_ID = "CapDimble/sentiment-monitoring-model"  # Pretrained model on Hugging Face
LOG_DIR = "data/monitoring"
LOG_FILE = os.path.join(LOG_DIR, "sentiment_log.csv")
TREND_PLOT = os.path.join(LOG_DIR, "sentiment_trend.png")


# Data and analysis functions
def get_new_posts():
    """Return a simulated list of new social media posts."""
    return [
        "I love the new AI tool from MachineInnovators!",
        "The product update was terrible, full of bugs.",
        "Support team was great today!",
        "Nothing special about the latest release.",
        "Awful user experience again...",
    ]


def analyze_sentiment(posts, model_id=MODEL_ID):
    """Run a sentiment analysis model on given text posts."""
    classifier = pipeline("sentiment-analysis", model=model_id)
    preds = classifier(posts)
    df = pd.DataFrame(preds)
    df["text"] = posts
    df["timestamp"] = datetime.now()
    return df


def log_results(df):
    """Append analyzed sentiment results to a CSV log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
    print(f"Logged {len(df)} new entries to {LOG_FILE}")


def show_summary():
    """Display the most recent sentiment log entries."""
    df = pd.read_csv(LOG_FILE)
    print("\n=== Current Sentiment Log ===")
    print(df.tail())
    return df


def plot_sentiment_trend(df):
    """Plot and save a simple sentiment distribution chart."""
    if "label" not in df.columns:
        print("No 'label' column found in logs. Skipping plot.")
        return

    sent_dist = df["label"].value_counts(normalize=True)

    plt.figure(figsize=(5, 3))
    plt.bar(sent_dist.index, sent_dist.values, color="skyblue")
    plt.title("Sentiment Distribution Over Time")
    plt.ylabel("Proportion")
    plt.tight_layout()

    plt.savefig(TREND_PLOT)
    plt.show()
    print(f"Sentiment trend saved to {TREND_PLOT}")
