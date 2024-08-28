from mapping.feature_columns import FEATURES_COLUMNS
from models.logistic_regression import create_log_regression_model
from models.random_forest import create_random_forest_model
from process.output_results import create_model_outputs
from process.process_features import create_df_all_features
from utils.load import load_model_pickle, save_model_pickle

PATH_DATA = "X_trainToronto.csv"
PATH_REVIEWS = "reviewsTrainToronto.csv"


def main():
    df_final = create_df_all_features(PATH_DATA, PATH_REVIEWS)

    model_log_regression = create_log_regression_model(
        df_final, FEATURES_COLUMNS, target_column="destaque"
    )
    # save_model_pickle(model_log_regression, "model_log_regression.pkl")
    # model = load_model_pickle("model_log_regression.pkl")
    create_model_outputs(model_log_regression, "log_reg")

    model_random_forest = create_random_forest_model(
        df_final, FEATURES_COLUMNS, target_column="destaque"
    )
    # save_model_pickle(model_random_forest, "model_random_forest.pkl")
    # model = load_model_pickle("model_random_forest.pkl")
    create_model_outputs(model_random_forest, "random_forest")


if __name__ == "__main__":
    main()
