import pandas as pd
import joblib

from prophet import Prophet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def load_data(file_path):
    """
    Load dataset
    """

    df = pd.read_csv(file_path)

    return df


def prepare_state_data(df, state_name):
    """
    Prepare one state dataset
    """

    state_df = df[df['State'] == state_name].copy()

    # Convert date
    state_df['Date'] = pd.to_datetime(
        state_df['Date']
    )

    # Sort data
    state_df = state_df.sort_values(
        'Date'
    )

    # Prophet format
    prophet_df = state_df[['Date', 'Sales']]

    prophet_df.columns = ['ds', 'y']

    return prophet_df


def train_test_split(df):
    """
    Time series split
    """

    split_index = int(len(df) * 0.8)

    train = df[:split_index]

    test = df[split_index:]

    return train, test


def train_prophet_model(train):
    """
    Train Prophet model
    """

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train)

    return model


def evaluate_model(model, test):
    """
    Evaluate Prophet model
    """

    future = model.make_future_dataframe(
        periods=len(test)
    )

    forecast = model.predict(future)

    predictions = forecast['yhat'].tail(
        len(test)
    ).values

    mae = mean_absolute_error(
        test['y'],
        predictions
    )

    rmse = mean_squared_error(
        test['y'],
        predictions
    ) ** 0.5

    print("\nProphet Performance")

    print(f"MAE: {mae:,.2f}")

    print(f"RMSE: {rmse:,.2f}")

    return predictions


def save_model(model, output_path):
    """
    Save Prophet model
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

    # Prepare data
    prophet_df = prepare_state_data(
        df,
        state_name
    )

    print("\nDataset Shape:")

    print(prophet_df.shape)

    # Split dataset
    train, test = train_test_split(
        prophet_df
    )

    print("\nTrain Shape:")

    print(train.shape)

    print("\nTest Shape:")

    print(test.shape)

    # Train model
    model = train_prophet_model(
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
        "saved_models/prophet_model.pkl"
    )

    print("\nSample Predictions:")

    print(predictions[:5])