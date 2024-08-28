import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def transform_to_lower(df):
    df["text"] = df["text"].str.lower()

    return df


def remove_ponctuation(df):
    df["text"] = df["text"].str.replace("[^\w\s]", "", regex=True)

    return df


def apply_tokenization(df):
    df["tokens"] = df["text"].str.split()

    return df


def remove_stop_words(df):
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    df["text"] = df["tokens"].apply(
        lambda x: " ".join(word for word in x if word not in stop_words)
    )

    return df


def apply_lemmatization(df):
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()

    df["text"] = df["text"].apply(
        lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split())
    )
    return df


def preprocessing_text(df):
    df = transform_to_lower(df)

    df = remove_ponctuation(df)

    df = apply_tokenization(df)

    df = remove_stop_words(df)

    # df = apply_stemming_text(df)
    df = apply_lemmatization(df)

    return df
