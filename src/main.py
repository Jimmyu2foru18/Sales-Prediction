import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing.data_loader import load_data
from data_processing.feature_engineering import create_features
from model.data_preprocessor import preprocess_data
from model.sales_predictor import train_model, evaluate_model

def main():
    # Define the path to the dataset
    data_filepath = "data/bigmart.csv"

    try:
        # 1. Load Data
        print("Loading data...")
        df = load_data(data_filepath)
        if df is None:
            print("Failed to load data. Exiting.")
            return

        # 2. Feature Engineering
        print("\nPerforming feature engineering...")
        df_with_features = create_features(df)
        if df_with_features is None:
            print("Feature engineering failed. Exiting.")
            return

        # 3. Preprocess Data
        print("\nPreprocessing data...")
        X_processed, y, preprocessor = preprocess_data(df_with_features)
        if X_processed is None or y is None:
            print("Data preprocessing failed. Exiting.")
            return

        # 4. Split Data
        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")

        # 5. Train and Evaluate Multiple Models
        models = ['linear', 'ridge', 'lasso', 'rf']
        best_model = None
        best_rmse = float('inf')
        best_model_type = None

        print("\nTraining and evaluating multiple models...")
        for model_type in models:
            print(f"\nTraining {model_type} model...")
            model, cv_scores = train_model(X_train, y_train, model_type=model_type)
            
            if model is not None:
                print(f"\nEvaluating {model_type} model...")
                metrics = evaluate_model(model, X_test, y_test)
                
                if metrics and metrics['RMSE'] < best_rmse:
                    best_rmse = metrics['RMSE']
                    best_model = model
                    best_model_type = model_type

        if best_model is not None:
            print(f"\nBest performing model: {best_model_type}")
            print(f"Best RMSE: {best_rmse:.2f}")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()