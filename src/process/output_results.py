from src.process.apply_test_data import apply_model_test_data
from utils.load import save_final_df


def create_model_outputs(model, model_name, csv_file_name):
    business_id, y_pred = apply_model_test_data(model, model_name)

    save_final_df(business_id, y_pred, file_name=csv_file_name)
