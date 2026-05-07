import json


def compare_models():
    """
    Compare forecasting model performances
    """

    results = {
        "LSTM": 3380128.74,
        "XGBoost": 9543993.02,
        "Prophet": 23290837.73,
        "SARIMA": 27051901.52
    }

    print("\nModel Performance Comparison\n")

    for model, rmse in results.items():

        print(f"{model}: {rmse:,.2f}")

    # Select best model
    best_model = min(
        results,
        key=results.get
    )

    best_rmse = results[best_model]

    print("\nBest Model Selected")

    print(f"Model: {best_model}")

    print(f"RMSE: {best_rmse:,.2f}")

    return results, best_model, best_rmse


def save_model_registry(
    results,
    best_model,
    best_rmse
):
    """
    Save model registry
    """

    registry = {
        "best_model": best_model,
        "best_rmse": best_rmse,
        "all_models": results
    }

    with open(
        "saved_models/model_registry.json",
        "w"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )

    print(
        "\nModel registry saved to:"
    )

    print(
        "saved_models/model_registry.json"
    )


if __name__ == "__main__":

    results, best_model, best_rmse = (
        compare_models()
    )

    save_model_registry(
        results,
        best_model,
        best_rmse
    )