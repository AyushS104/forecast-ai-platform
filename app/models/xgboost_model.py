import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor


def load_data(file_path):
    """
    Load feature engineered dataset
    """

    df = pd.read_csv(file_path)

    return df


def prepare_features(df):
    """
    Select features and target
    """

    features = [
        'lag_1',
        'lag_7',
        'lag_30',
        'rolling_mean_7',
        'rolling_std_7',
        'day_of_week',
        'month',
        'week_of_year',
        'holiday_flag'
    ]

    X = df[features]

    y = df['Sales']

    return X, y


def train_test_split_time_series(X, y):
    """
    Time series split
    """

    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train):
    """
    Train XGBoost model
    """

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    print("\nModel Performance")

    print(f"MAE: {mae:,.2f}")

    print(f"RMSE: {rmse:,.2f}")

    return predictions


def save_model(model, output_path):
    """
    Save trained model
    """

    joblib.dump(model, output_path)

    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":

    # Load data
    df = load_data(
        "data/processed/feature_engineered_sales.csv"
    )

    print("\nDataset Shape:")
    print(df.shape)

    # Prepare features
    X, y = prepare_features(df)

    # Train test split
    X_train, X_test, y_train, y_test = (
        train_test_split_time_series(X, y)
    )

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)

    # Train model
    model = train_xgboost_model(
        X_train,
        y_train
    )

    # Evaluate model
    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save model
    save_model(
        model,
        "saved_models/xgboost_model.pkl"
    )

    print("\nSample Predictions:")

    print(predictions[:5])