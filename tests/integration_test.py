import os
import pandas as pd
import pytest
from transformers import pipeline
from src.train import load_data, tokenize_data

DATA_PATH = "data/raw/social_sentiment.csv"
MODEL_DIR = "models/roberta_sentiment_model"
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

ALLOWED_LABELS = {"positive", "neutral", "negative", 0, 1, 2}


@pytest.mark.order(1)
def test_data_loading_and_preprocessing():
    """Test dataset loading, preprocessing, and label consistency."""
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    df = load_data(DATA_PATH)
    
    assert {"text", "label"}.issubset(df.columns)
    assert len(df) >= 3
    assert df["text"].isna().sum() == 0

    # Check label validity (handles both numeric and string encodings)
    label_values = df["label"].unique().tolist() if "label" in df.columns else df["sentiment"].unique().tolist()
    allowed = {"positive", "neutral", "negative"} if all(isinstance(x, str) for x in label_values) else {0, 1, 2}
    assert set(label_values).issubset(allowed), f"Unexpected labels: {set(label_values)}"


@pytest.mark.order(2)
def test_tokenization_output():
    """Verify that tokenization produces required model input fields."""
    df = load_data(DATA_PATH)
    tok_ds = tokenize_data(df.head(3))
    sample = tok_ds[0]
    for key in ["input_ids", "attention_mask", "label"]:
        assert key in sample, f"Missing key '{key}' in tokenized sample"


@pytest.mark.order(3)
def test_inference_with_model():
    """Ensure model inference runs successfully using local or fallback model."""
    model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else HF_MODEL
    print(f"Using model from: {model_path}")

    df = load_data(DATA_PATH).head(5)
    texts = df["text"].tolist()

    classifier = pipeline("sentiment-analysis", model=model_path)
    preds = classifier(texts)

    assert len(preds) == len(texts)
    for pred in preds:
        assert "label" in pred and "score" in pred
        assert pred["label"] in ALLOWED_LABELS