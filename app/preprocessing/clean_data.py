import pandas as pd
import holidays


def load_clean_data(file_path):
    """
    Load cleaned dataset
    """

    df = pd.read_csv(file_path)

    return df


def convert_date(df):
    """
    Convert date column to datetime
    """

    df['Date'] = pd.to_datetime(df['Date'])

    return df


def create_lag_features(df):
    """
    Create lag features
    """

    df['lag_1'] = df.groupby('State')['Sales'].shift(1)

    df['lag_7'] = df.groupby('State')['Sales'].shift(7)

    df['lag_30'] = df.groupby('State')['Sales'].shift(30)

    return df


def create_rolling_features(df):
    """
    Create rolling mean and rolling std features
    """

    df['rolling_mean_7'] = (
        df.groupby('State')['Sales']
        .transform(lambda x: x.rolling(7).mean())
    )

    df['rolling_std_7'] = (
        df.groupby('State')['Sales']
        .transform(lambda x: x.rolling(7).std())
    )

    return df


def create_date_features(df):
    """
    Create date-based features
    """

    df['day_of_week'] = df['Date'].dt.dayofweek

    df['month'] = df['Date'].dt.month

    df['week_of_year'] = (
        df['Date']
        .dt
        .isocalendar()
        .week
    )

    return df


def create_holiday_feature(df):
    """
    Create holiday feature
    """

    india_holidays = holidays.India()

    df['holiday_flag'] = df['Date'].apply(
        lambda x: 1 if x in india_holidays else 0
    )

    return df


def drop_missing_rows(df):
    """
    Remove rows with null values
    created by lagging operations
    """

    df = df.dropna()

    return df


def save_feature_engineered_data(df, output_path):
    """
    Save processed dataset
    """

    df.to_csv(output_path, index=False)

    print(f"Feature engineered data saved to: {output_path}")


if __name__ == "__main__":

    # Load cleaned data
    df = load_clean_data(
        "data/processed/cleaned_sales.csv"
    )

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # Convert date
    df = convert_date(df)

    # Create lag features
    df = create_lag_features(df)

    # Create rolling statistics
    df = create_rolling_features(df)

    # Create date features
    df = create_date_features(df)

    # Create holiday feature
    df = create_holiday_feature(df)

    # Remove null rows
    df = drop_missing_rows(df)

    print("\nFeature Engineered Dataset Shape:")
    print(df.shape)

    # Save feature engineered dataset
    save_feature_engineered_data(
        df,
        "data/processed/feature_engineered_sales.csv"
    )

    print("\nFeature Engineered Data Preview:")
    print(df.head())

    print("\nColumns:")
    print(df.columns)