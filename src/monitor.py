import sys, os
# Add project root (one level up) to PYTHONPATH 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.monitor_utils import get_new_posts, analyze_sentiment, log_results, show_summary, plot_sentiment_trend

if __name__ == "__main__":
    df = analyze_sentiment(get_new_posts())
    log_results(df)
    show_summary()
    plot_sentiment_trend(df)
