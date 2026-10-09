"""
Tweet Sentiment Analysis using VADER.
Accepts a CSV with a 'text' column or runs with built-in sample sentences.
VADER is a rule/lexicon-based sentiment analyzer, not a trained custom model.
"""
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    """Remove URLs and normalize whitespace while retaining sentiment cues."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_sentiment(compound: float) -> str:
    """Apply common VADER compound-score thresholds."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_texts(texts: pd.Series) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for original in texts.fillna("").astype(str):
        cleaned = clean_text(original)
        scores = analyzer.polarity_scores(cleaned)
        rows.append({
            "text": original,
            "cleaned_text": cleaned,
            "positive_score": scores["pos"],
            "neutral_score": scores["neu"],
            "negative_score": scores["neg"],
            "compound_score": scores["compound"],
            "sentiment": classify_sentiment(scores["compound"]),
        })
    return pd.DataFrame(rows)


def load_texts(csv_path: str | None) -> pd.Series:
    if csv_path:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a column named 'text'.")
        return df["text"]
    return pd.Series([
        "I love this new phone, the camera is amazing!",
        "The app keeps crashing and customer support was terrible.",
        "The package arrived today.",
        "Really happy with the service 😊",
        "I am disappointed. This was a waste of money.",
        "The update is okay, nothing special.",
    ], name="text")


def main(csv_path: str | None) -> None:
    texts = load_texts(csv_path)
    if texts.empty:
        raise ValueError("No tweets found in the input file.")

    results = analyze_texts(texts)
    output_csv = OUTPUT_DIR / "sentiment_results.csv"
    results.to_csv(output_csv, index=False)

    counts = results["sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )
    print("\nTweet Sentiment Analysis")
    print("=" * 30)
    print(f"Tweets analyzed: {len(results)}")
    print("\nSentiment counts:")
    print(counts.to_string())
    print(f"\nSaved results to: {output_csv}")

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of tweets")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "sentiment_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved chart to: {chart_path}")

    print("\nSample classifications:")
    for _, row in results.head(5).iterrows():
        print(f"- [{row['sentiment']}] {row['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify tweet text as positive, neutral, or negative.")
    parser.add_argument("--csv", default=None, help="Optional CSV file containing a 'text' column")
    args = parser.parse_args()
    try:
        main(args.csv)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")
