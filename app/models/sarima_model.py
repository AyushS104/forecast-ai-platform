import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data(file_path):
    """
    Load dataset
    """

    df = pd.read_csv(file_path)

    return df


def prepare_state_data(df, state_name):
    """
    Filter one state only
    """

    state_df = df[df['State'] == state_name].copy()

    # Convert date column
    state_df['Date'] = pd.to_datetime(
        state_df['Date']
    )

    # Sort by date
    state_df = state_df.sort_values(
        'Date'
    )

    # Set date as index
    state_df = state_df.set_index(
        'Date'
    )

    # Set daily frequency
    state_df = state_df.asfreq('D')

    return state_df


def train_test_split(state_df):
    """
    Time series split
    """

    split_index = int(len(state_df) * 0.8)

    train = state_df[:split_index]

    test = state_df[split_index:]

    return train, test


def train_sarima_model(train):
    """
    Train SARIMA model
    """

    model = SARIMAX(
        train['Sales'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    results = model.fit(
        disp=False
    )

    return results


def evaluate_model(model, test):
    """
    Evaluate SARIMA model
    """

    predictions = model.forecast(
        steps=len(test)
    )

    mae = mean_absolute_error(
        test['Sales'],
        predictions
    )

    rmse = mean_squared_error(
        test['Sales'],
        predictions
    ) ** 0.5

    print("\nSARIMA Performance")

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

    # Load dataset
    df = load_data(
        "data/processed/cleaned_sales.csv"
    )

    print("\nAvailable States:")

    print(df['State'].unique()[:10])

    # Select state
    state_name = "Alabama"

    # Prepare state data
    state_df = prepare_state_data(
        df,
        state_name
    )

    print("\nState Dataset Shape:")

    print(state_df.shape)

    # Train test split
    train, test = train_test_split(
        state_df
    )

    print("\nTrain Shape:")

    print(train.shape)

    print("\nTest Shape:")

    print(test.shape)

    # Train model
    model = train_sarima_model(
        train
    )

    # Evaluate model
    predictions = evaluate_model(
        model,
        test
    )

    # Save model
    save_model(
        model,
        "saved_models/sarima_model.pkl"
    )

    print("\nSample Predictions:")

    print(predictions[:5])