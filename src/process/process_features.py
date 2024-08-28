import pandas as pd

from process.aggregate import create_aggregated_reviews
from process.clean_data import clean_data
from process.preprocessing_reviews import preprocessing_text
from src.process.extract_features import extract_features
from src.process.reviews_features import apply_sentiment_analysis


def create_df_all_features(path_data, path_reviews):
    # Extract standard data features
    df_data = pd.read_csv(path_data)

    df_cleaned = clean_data(df_data)

    df_features_extracted = extract_features(df_cleaned)

    # Extract reviews features
    df_reviews = pd.read_csv(path_reviews)

    df_review_preprocessed = preprocessing_text(df_reviews)

    df_reviews_features_extracted = apply_sentiment_analysis(df_review_preprocessed)
    # df_reviews_features_extracted.to_csv("df_reviews_features_extracted.csv")
    # df_reviews_features_extracted = pd.read_csv("df_reviews_features_extracted.csv")

    df_aggregated_reviews_features = create_aggregated_reviews(df_reviews_features_extracted)

    df_final = pd.merge(
        df_features_extracted, df_aggregated_reviews_features, on="business_id", how="inner"
    )

    return df_final
