import json
import re
from ast import literal_eval


def categories_to_str(df):
    df["categories"] = df["categories"].fillna("").astype(str)

    return df


def clean_attribute_string(attr_str):
    attr_str = re.sub(r"u'([^']*)'", r"'\1'", attr_str)
    attr_str = re.sub(r"\\'", r"'", attr_str)
    return attr_str


def apply_literal_eval(val):
    if isinstance(val, dict):
        return val

    try:
        cleaned_val = clean_attribute_string(val)
        return literal_eval(cleaned_val)
    except (ValueError, SyntaxError):
        return {}


def parse_hours(hours_str):
    if isinstance(hours_str, dict):
        hours_dict = hours_str

        return hours_dict

    if isinstance(hours_str, float):
        return {}

    try:
        hours_dict = json.loads(hours_str.replace("'", '"'))
        return hours_dict
    except (ValueError, TypeError):
        return {}


def clean_data(df):
    df["address"].fillna("Unknown", inplace=True)
    df["postal_code"].fillna("Unknown", inplace=True)

    df["attributes"].fillna("{}", inplace=True)
    df["attributes"] = df["attributes"].apply(apply_literal_eval)

    df["hours"] = df["hours"].apply(parse_hours)
    df["hours_missing"] = df["hours"].isnull().astype(int)

    df = categories_to_str(df)

    return df
