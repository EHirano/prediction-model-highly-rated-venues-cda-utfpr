import pickle

import pandas as pd


def save_final_df(business_ids, y_pred, file_name):

    df_results = pd.DataFrame({"business_id": business_ids, "destaque": y_pred})

    df_results.to_csv(file_name, index=False)


def save_model_pickle(model, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def load_model_pickle(file_name):
    with open(file_name, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model
