import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Preprocesses the data for model training.

    Args:
        df (pandas.DataFrame): The input DataFrame with features.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The preprocessed features (X).
            - pandas.Series: The target variable (y).
            - ColumnTransformer: The preprocessor object.
    """
    if df is None:
        return None, None, None

    # Separate target variable
    X = df.drop('Item_Outlet_Sales', axis=1)
    y = df['Item_Outlet_Sales']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['number']).columns.drop(['Item_MRP', 'Outlet_Years'])

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like Item_MRP, Outlet_Years) as is for now
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Convert processed data back to DataFrame (optional, but can be helpful)
    # Note: Column names will be generic after one-hot encoding
    # X_processed_df = pd.DataFrame(X_processed)

    print("Data preprocessing completed.")
    return X_processed, y, preprocessor

if __name__ == "__main__":
    # Example usage (requires a sample DataFrame with features)
    data = {'Item_Identifier': ['FDA15', 'DRC01', 'FDN15', 'FDX07', 'NCD19'],
            'Item_Weight': [9.3, 5.92, 17.5, 19.2, 8.93],
            'Item_Fat_Content': ['Low Fat', 'Regular', 'Low Fat', 'Low Fat', 'Low Fat'],
            'Item_Visibility': [0.016, 0.019, 0.017, 0.000, 0.000],
            'Item_Type': ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household'],
            'Item_MRP': [249.8, 48.26, 141.6, 182.1, 53.86],
            'Outlet_Identifier': ['OUT049', 'OUT018', 'OUT049', 'OUT010', 'OUT013'],
            'Outlet_Establishment_Year': [1999, 2009, 1999, 1998, 1987],
            'Outlet_Size': ['Medium', 'Medium', 'Medium', 'Small', 'High'],
            'Outlet_Location_Type': ['Tier 1', 'Tier 3', 'Tier 1', 'Tier 3', 'Tier 3'],
            'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type1', 'Grocery Store', 'Supermarket Type1'],
            'Item_Outlet_Sales': [3735.1, 443.4, 2097.2, 732.3, 994.7],
            'Item_MRP_Log': [5.52, 3.88, 4.95, 5.20, 3.99],
            'Outlet_Years': [25, 15, 25, 26, 37]}
    sample_df = pd.DataFrame(data)

    X_processed, y, preprocessor = preprocess_data(sample_df)

    if X_processed is not None:
        print("\nPreprocessed features (first 5 rows):")
        print(X_processed[:5])
        print("\nTarget variable (first 5 values):")
        print(y.head())
        print("\nPreprocessor object created.")