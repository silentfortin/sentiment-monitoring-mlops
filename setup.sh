#!/bin/bash
# Setup script for Sentiment Monitoring MLOps Project (Windows)

echo "Setting up environment for sentiment-monitoring-mlops..."

# 1. Create and activate virtual environment
python -m venv .venv > /dev/null 2>&1
source .venv/Scripts/activate
echo "Virtual environment created and activated."

# 2. Upgrade pip (quiet mode)
python -m pip install --upgrade pip -q
echo "Pip upgraded."

# 3. Install dependencies (quiet mode)
pip install -r requirements.txt -q
echo "Dependencies installed."

# 4. Download RoBERTa model weights (optional)
echo "Downloading RoBERTa model weights..."
python - <<EOF
from transformers import AutoModelForSequenceClassification, AutoTokenizer
AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
EOF
echo "RoBERTa model downloaded."

# 5. Completion message
echo "Setup completed successfully."
echo "To activate your environment later, run:"
echo "source .venv/Scripts/activate"