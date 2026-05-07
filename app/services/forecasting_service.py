import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler


def load_dataset():
    """
    Load cleaned dataset
    """

    df = pd.read_csv(
        "data/processed/cleaned_sales.csv"
    )

    return df


def prepare_state_data(df, state_name):
    """
    Filter one state
    """

    state_df = df[
        df['State'] == state_name
    ].copy()

    state_df = state_df.sort_values(
        'Date'
    )

    return state_df


def scale_sales_data(state_df):
    """
    Scale sales data
    """

    scaler = MinMaxScaler()

    sales_data = state_df[
        ['Sales']
    ].values

    scaled_data = scaler.fit_transform(
        sales_data
    )

    return scaler, scaled_data


def load_lstm_model():
    """
    Load trained LSTM model
    """

    model = load_model(
        "saved_models/lstm_model.keras"
    )

    return model


def generate_forecast(
    state_name,
    forecast_days=56
):
    """
    Generate future forecast
    """

    # Load dataset
    df = load_dataset()

    # Prepare state data
    state_df = prepare_state_data(
        df,
        state_name
    )

    # Scale data
    scaler, scaled_data = (
        scale_sales_data(state_df)
    )

    # Load trained model
    model = load_lstm_model()

    # Last 30 days
    window_size = 30

    last_sequence = scaled_data[
        -window_size:
    ]

    forecast = []

    current_sequence = (
        last_sequence.copy()
    )

    # Predict future values
    for _ in range(forecast_days):

        X_input = np.array(
            [current_sequence]
        )

        prediction = model.predict(
            X_input,
            verbose=0
        )

        forecast.append(
            prediction[0][0]
        )

        current_sequence = np.vstack(
            [
                current_sequence[1:],
                prediction
            ]
        )

    # Convert back to original scale
    forecast = np.array(
        forecast
    ).reshape(-1, 1)

    forecast = scaler.inverse_transform(
        forecast
    )

    # Convert to list
    forecast = forecast.flatten().tolist()

    return forecast