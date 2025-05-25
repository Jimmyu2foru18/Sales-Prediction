import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_model(X_train, y_train, model_type='linear'):
    """
    Trains a regression model with cross-validation.

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target variable.
        model_type (str): Type of model to train ('linear', 'ridge', 'lasso', 'rf').

    Returns:
        tuple: (trained_model, cv_scores)
    """
    if X_train is None or y_train is None:
        return None, None

    try:
        # Select model based on type
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"Cross-validation RMSE scores: {cv_rmse}")
        print(f"Average CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")

        # Train final model on full training data
        model.fit(X_train, y_train)
        print(f"Model training completed using {model_type} regression.")
        
        return model, cv_scores

    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using multiple metrics.

    Args:
        model: The trained model.
        X_test (numpy.ndarray or pandas.DataFrame): Testing features.
        y_test (pandas.Series): Testing target variable.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if model is None or X_test is None or y_test is None:
        return None

    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        print("\nModel Evaluation Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2 Score: {r2:.3f}")
        print(f"MAPE: {mape:.2f}%")

        return metrics

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

if __name__ == "__main__":
    # Example usage (requires preprocessed data)
    # This is a simplified example; in a real scenario, you'd load and preprocess data first
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_model = train_model(X_train, y_train)

    if trained_model is not None:
        evaluation_metrics = evaluate_model(trained_model, X_test, y_test)
        if evaluation_metrics:
            print("\nEvaluation Metrics:")
            print(evaluation_metrics)