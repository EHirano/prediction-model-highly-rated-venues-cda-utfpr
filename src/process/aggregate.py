def create_aggregated_reviews(df):
    """Creates statistical data based on the sentiment analysis, grouping them by business_id"""

    df_reviews_aggregated = df.groupby("business_id").agg(
        {
            "useful": ["mean", "median", "std"],
            "funny": ["mean", "median", "std"],
            "cool": ["mean", "median", "std"],
            "polarity": ["mean", "median", "std"],
            "subjectivity": ["mean", "median", "std"],
            "vader_neg": ["mean", "median", "std"],
            "vader_neu": ["mean", "median", "std"],
            "vader_pos": ["mean", "median", "std"],
            "vader_compound": ["mean", "median", "std"],
        }
    )

    df_reviews_aggregated.columns = [
        "_".join(col).strip() for col in df_reviews_aggregated.columns.values
    ]

    df_reviews_aggregated = df_reviews_aggregated.reset_index()
    return df_reviews_aggregated
