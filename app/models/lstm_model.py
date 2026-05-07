import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input
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

    # Sort by date
    state_df = state_df.sort_values(
        'Date'
    )

    # Select sales column
    sales_data = state_df[['Sales']].values

    return sales_data


def scale_data(data):
    """
    Scale sales data
    """

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        data
    )

    return scaler, scaled_data


def create_sequences(data, window_size=30):
    """
    Create LSTM sequences
    """

    X = []

    y = []

    for i in range(window_size, len(data)):

        X.append(
            data[i-window_size:i]
        )

        y.append(data[i])

    X = np.array(X)

    y = np.array(y)

    return X, y


def train_test_split(X, y):
    """
    Time series split
    """

    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]

    X_test = X[split_index:]

    y_train = y[:split_index]

    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def build_lstm_model(input_shape):
    """
    Build LSTM model
    """

    model = Sequential()

    # Input layer
    model.add(
        Input(shape=input_shape)
    )

    # First LSTM layer
    model.add(
        LSTM(
            64,
            return_sequences=True
        )
    )

    # Second LSTM layer
    model.add(
        LSTM(32)
    )

    # Output layer
    model.add(
        Dense(1)
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    scaler
):
    """
    Evaluate LSTM model
    """

    predictions = model.predict(
        X_test
    )

    # Convert predictions back
    predictions = scaler.inverse_transform(
        predictions
    )

    # Convert actual values back
    actual = scaler.inverse_transform(
        y_test
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = mean_squared_error(
        actual,
        predictions
    ) ** 0.5

    print("\nLSTM Performance")

    print(f"MAE: {mae:,.2f}")

    print(f"RMSE: {rmse:,.2f}")

    return predictions


def save_model(model):
    """
    Save trained model
    """

    model.save(
        "saved_models/lstm_model.keras"
    )

    print(
        "\nModel saved to: "
        "saved_models/lstm_model.keras"
    )


if __name__ == "__main__":

    # Load dataset
    df = load_data(
        "data/processed/cleaned_sales.csv"
    )

    # Select state
    state_name = "Alabama"

    # Prepare sales data
    sales_data = prepare_state_data(
        df,
        state_name
    )

    print("\nSales Data Shape:")

    print(sales_data.shape)

    # Scale data
    scaler, scaled_data = scale_data(
        sales_data
    )

    # Create sequences
    X, y = create_sequences(
        scaled_data
    )

    print("\nSequence Shape:")

    print(X.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y)
    )

    print("\nTrain Shape:")

    print(X_train.shape)

    print("\nTest Shape:")

    print(X_test.shape)

    # Build model
    model = build_lstm_model(
        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32
    )

    # Evaluate model
    predictions = evaluate_model(
        model,
        X_test,
        y_test,
        scaler
    )

    # Save model
    save_model(model)

    print("\nSample Predictions:")

    print(predictions[:5])