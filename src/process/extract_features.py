from ast import literal_eval

import pandas as pd


def extract_attributes(attrs, key):
    if key in attrs:
        return attrs[key]
    else:
        return None


def extract_all_attributes(df):
    attributes_keys = [
        "RestaurantsPriceRange2",
        "OutdoorSeating",
        "BusinessParking",
        "GoodForKids",
        "RestaurantsReservations",
    ]

    for key in attributes_keys:
        df[key] = df["attributes"].apply(lambda x: extract_attributes(x, key))

    return df


def convert_to_boolean(df):
    boolean_columns = ["OutdoorSeating", "GoodForKids", "RestaurantsReservations"]
    for col in boolean_columns:
        df[col] = df[col].replace({"True": 1, "False": 0, "None": 0, None: 0}).fillna(0).astype(int)

    return df


def calculate_total_hours(hours):
    total = 0
    for day, hours_range in hours.items():
        if hours_range != "None":
            open_time, close_time = hours_range.split("-")
            total += int(close_time.split(":")[0]) - int(open_time.split(":")[0])

    return total


def is_weekend_open(hours):
    weekend_days = ["Saturday", "Sunday"]

    for day in weekend_days:
        if day in hours and hours[day] != "None":
            return 1
    return 0


def has_category(categories, cat_name):
    if cat_name in categories:
        return 1
    else:
        return 0


def create_category_feature(df):
    categories_list = ["Restaurants", "Food", "Shopping", "Nightlife"]

    for cat in categories_list:
        df[cat] = df["categories"].apply(lambda x: has_category(x, cat))

    return df


def drop_att_hours(df):
    """Depois de extrair as features das colunas, essas podem ser descartadas para o modelo"""
    df.drop(columns=["attributes", "hours"], inplace=True)

    return df


def extract_parking_features(parking_info):
    parking_features = {
        "parking_garage": 0,
        "parking_street": 0,
        "parking_validated": 0,
        "parking_lot": 0,
        "parking_valet": 0,
    }

    if parking_info is None or not isinstance(parking_info, str):
        # Returns all features as 0 if it doesn't have the data associated - this could need a reavaluation
        # since it can impact the model's performance if the feature has a high impact on the predicted feature
        return pd.Series(parking_features)

    try:
        if isinstance(parking_info, str):
            parking_dict = literal_eval(parking_info.replace("'", '"'))
            # print(parking_dict)

            if parking_dict:
                parking_features["parking_garage"] = 1 if parking_dict.get("garage", False) else 0
                parking_features["parking_street"] = 1 if parking_dict.get("street", False) else 0
                parking_features["parking_validated"] = (
                    1 if parking_dict.get("validated", False) else 0
                )
                parking_features["parking_lot"] = 1 if parking_dict.get("lot", False) else 0
                parking_features["parking_valet"] = 1 if parking_dict.get("valet", False) else 0
            else:
                # Case when it has a "None" string - returns all features as 0
                return pd.Series(parking_features)
    except (ValueError, SyntaxError) as e:
        # print(parking_dict)
        print(e)
        pass

    return pd.Series(parking_features)


def extract_features(df):
    df_extracted_att = extract_all_attributes(df)

    df = convert_to_boolean(df_extracted_att)

    df["RestaurantsPriceRange2"] = pd.to_numeric(
        df["RestaurantsPriceRange2"], errors="coerce"
    ).fillna(0)

    df["total_hours"] = df["hours"].apply(calculate_total_hours)

    df["weekend_open"] = df["hours"].apply(is_weekend_open)

    df_cat_feat = create_category_feature(df)

    df_parking_features = df_cat_feat["BusinessParking"].apply(extract_parking_features)

    df_concat = pd.concat([df_cat_feat, df_parking_features], axis=1)

    df_final = drop_att_hours(df_concat)

    return df_final
