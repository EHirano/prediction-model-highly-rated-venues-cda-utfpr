from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate_model(model, X_test_scaled_1, y_test_1):
    y_pred = model.predict(X_test_scaled_1)

    accuracy = accuracy_score(y_test_1, y_pred)
    precision = precision_score(y_test_1, y_pred)
    recall = recall_score(y_test_1, y_pred)
    f1 = f1_score(y_test_1, y_pred)
    conf_matrix = confusion_matrix(y_test_1, y_pred)
    class_report = classification_report(y_test_1, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")


def create_log_regression_model(
    df, features_columns: list, target_column="destaque", print_model_report=True
):
    try:
        print("Creating logistic regression model")
        X_1 = df[features_columns]
        y_1 = df[target_column]

        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
            X_1, y_1, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled_1 = scaler.fit_transform(X_train_1)
        X_test_scaled_1 = scaler.transform(X_test_1)

        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled_1, y_train_1)

        if print_model_report:
            evaluate_model(model, X_test_scaled_1, y_test_1)

        return model
    except Exception as e:
        print(f"Error creating linear regression model: {e}")
        raise e
