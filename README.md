# Twitter Sentiment Analysis

A simple NLP project that classifies tweet-like text as **Positive**, **Negative**, or **Neutral** using VADER sentiment analysis.

> This project analyzes text supplied by you. It does not automatically scrape Twitter/X or connect to an X account.

## Features
- Cleans URLs, mentions, and extra whitespace
- Calculates positive, neutral, negative, and compound sentiment scores
- Classifies each row using standard VADER compound thresholds
- Saves detailed results to CSV and a sentiment distribution chart

## Setup

Python 3.10+ recommended.

```bash
git clone <your-repository-url>
cd twitter-sentiment-analysis
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install and run the included sample:

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

Run it on your own CSV:

```bash
python sentiment_analysis.py --csv my_tweets.csv
```

Your CSV must have a column named `text`, with one tweet per row. See `sample_tweets.csv`.

## Outputs
- `outputs/sentiment_results.csv`
- `outputs/sentiment_distribution.png`

## How it works
VADER is a lexicon- and rule-based sentiment tool designed for social-media-style text. It returns a compound score from -1 to +1. This project labels scores >= 0.05 Positive, <= -0.05 Negative, and scores in between Neutral.

## Limitations
- Sarcasm, slang, context, mixed-language tweets, and domain-specific meanings may be misclassified.
- VADER is not a custom-trained model and is not a guarantee of human-level accuracy.
- Only use data you are permitted to process and respect the platform's terms and user privacy.

## Possible improvements
Evaluate against manually labelled data, add language detection, compare with NLTK or transformer models, and build a Streamlit interface.
