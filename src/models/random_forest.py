from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def evaluate_model(rf_model, X_test, y_test):
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")


def create_random_forest_model(
    df, features_columns: list, target_column="destaque", print_model_report=True
):
    try:
        X = df[features_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        if print_model_report:
            evaluate_model(rf_model, X_test, y_test)

        return rf_model
    except Exception as e:
        print(f"Error creating random forest model: {e}")
        raise e
