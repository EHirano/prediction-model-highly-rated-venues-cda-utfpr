import pandas as pd

from mapping.feature_columns import FEATURES_COLUMNS
from models.logistic_regression import create_log_regression_model
from process.clean_data import clean_data
from src.process.apply_test_data import apply_model_test_data
from src.process.extract_features import extract_features
from utils.load import save_final_df, save_model_pickle

PATH_DATA = "X_trainToronto.csv"

df = pd.read_csv("df_sentiment_analysis_swift_all_data.csv")

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

# Flatten the MultiIndex columns
df_reviews_aggregated.columns = [
    "_".join(col).strip() for col in df_reviews_aggregated.columns.values
]

# Reset the index so 'business_id' becomes a column again
df_reviews_aggregated = df_reviews_aggregated.reset_index()

df_data = pd.read_csv(PATH_DATA)

df_cleaned = clean_data(df_data)

df_features_extracted = extract_features(df_cleaned)

df_final = pd.merge(df_features_extracted, df_reviews_aggregated, on="business_id", how="inner")

model_log_regression = create_log_regression_model(
    df_final, FEATURES_COLUMNS, target_column="destaque"
)
save_model_pickle(model_log_regression, "model_agg.pkl")

business_id_1, y_pred_1 = apply_model_test_data(model_log_regression)

save_final_df(business_id_1, y_pred_1, file_name="df_test_agg_log_regression.csv")
