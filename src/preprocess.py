import re
import string
import pandas as pd

# Text cleaning utilities tailored for short social posts

# Precompiled regex patterns for performance
emoji_pattern = re.compile("[" 
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # Flags
    u"\u2702-\u27B0"
    u"\u24C2-\U0001F251"
    "]+", flags=re.UNICODE)

url_pattern = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))')
user_pattern = re.compile(r'@[^\s]+')
hashtag_pattern = re.compile(r'#([^\s]+)')
short_words_pattern = re.compile(r'\W*\b\w{1,3}\b')

def clean_text(text: str) -> str:
    """Basic text cleaning for social media content."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = emoji_pattern.sub("", text)
    text = url_pattern.sub("", text)
    text = user_pattern.sub("", text)
    text = hashtag_pattern.sub(r"\1", text)
    text = short_words_pattern.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(string.punctuation + " \t\n\r")
    return text

def preprocess_texts(texts):
    """Apply text cleaning to Series, list, or string input."""
    if isinstance(texts, pd.Series):
        return texts.fillna("").apply(clean_text)
    elif isinstance(texts, list):
        return [clean_text(t) for t in texts]
    elif isinstance(texts, str):
        return clean_text(texts)
    else:
        raise TypeError(f"Unsupported input type: {type(texts)}")