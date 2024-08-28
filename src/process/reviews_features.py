import pandas as pd
import swifter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from utils.timer import timer


def vader_sentiment(text):
    if not isinstance(text, str):
        # Lida com os casos de NaN data
        return 0.0, 0.0, 0.0, 0.0

    sid = SentimentIntensityAnalyzer()

    sentiment_dict = sid.polarity_scores(text)
    return (
        sentiment_dict["neg"],
        sentiment_dict["neu"],
        sentiment_dict["pos"],
        sentiment_dict["compound"],
    )


def get_sentiment(text):
    if not isinstance(text, str):
        # Lida com NaN data
        return 0.0, 0.0

    blob = TextBlob(text)

    return blob.sentiment.polarity, blob.sentiment.subjectivity


@timer
def apply_sentiment_analysis(df):
    """Creates the following features:
    - Using TextBlob: polarity, subjectivity;
    - Using VADER: neg, neu, pos, compound

    Process the text related data in parallel using the Swifter library."""

    print("Applying TextBlob sentiment analysis")
    df[["polarity", "subjectivity"]] = df["text"].swifter.apply(get_sentiment).apply(pd.Series)

    print("Applying Vader sentiment analysis")
    df[["vader_neg", "vader_neu", "vader_pos", "vader_compound"]] = (
        df["text"].swifter.apply(vader_sentiment).apply(pd.Series)
    )

    return df
