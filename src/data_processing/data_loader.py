import pandas as pd

def load_data(filepath):
    """
    Loads the BigMart sales data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
\    Returns:
        pandas.DataFrame: The loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    # Assuming bigmart.csv is in the 'data' directory relative to the project root
    data_path = "..\data\bigmart.csv"
    sales_data = load_data(data_path)

    if sales_data is not None:
        print("Data loaded successfully. First 5 rows:")
        print(sales_data.head())
        print("\nData Info:")
        sales_data.info()