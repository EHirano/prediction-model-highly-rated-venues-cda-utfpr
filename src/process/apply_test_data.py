from sklearn.preprocessing import StandardScaler

from mapping.feature_columns import FEATURES_COLUMNS
from process.process_features import create_df_all_features

PATH_TEST_DATA = "X_testToronto.csv"
PATH_TEST_REVIEWS = "reviewsTestToronto.csv"


def apply_model_test_data(model, model_type: str):

    df_final = create_df_all_features(PATH_TEST_DATA, PATH_TEST_REVIEWS)

    business_ids = df_final["business_id"]

    X_test = df_final[FEATURES_COLUMNS]

    if model_type == "log_reg":
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        y_pred = model.predict(X_test_scaled)

    elif model_type == "random_forest":
        y_pred = model.predict(X_test)

    return business_ids, y_pred
