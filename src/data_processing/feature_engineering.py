import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates new features from the raw sales data and handles missing values.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with new features and cleaned data.
    """
    if df is None:
        return None

    try:
        # Create copy to avoid modifying original data
        df = df.copy()

        # Handle missing values
        print("Handling missing values...")
        # Fill missing Item_Weight with median weight per Item_Type
        df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.median()))
        # If any remain missing, fill with overall median
        df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
        
        # Fill missing Outlet_Size based on Outlet_Type mode
        df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.fillna(x.mode()[0]))

        # Basic transformations
        print("Creating basic transformations...")
        df['Item_MRP_Log'] = np.log(df['Item_MRP'])
        df['Outlet_Years'] = 2024 - df['Outlet_Establishment_Year']

        # Item visibility features
        print("Processing visibility features...")
        df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())
        df['Item_Visibility_Normalized'] = df.groupby('Item_Type')['Item_Visibility'].transform(lambda x: x / x.mean())

        # Price features
        print("Creating price-related features...")
        df['Price_Per_Weight'] = df['Item_MRP'] / df['Item_Weight']
        df['Item_MRP_Normalized'] = df.groupby('Item_Type')['Item_MRP'].transform(lambda x: x / x.mean())

        # Categorical feature encoding
        print("Encoding categorical features...")
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'LF': 'Low Fat', 
            'reg': 'Regular', 
            'low fat': 'Low Fat',
            'Regular': 'Regular',
            'Low Fat': 'Low Fat'
        })
        
        # Store performance indicators
        print("Calculating store performance indicators...")
        df['Store_Score'] = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].transform('mean')
        df['Store_Score_Normalized'] = df['Store_Score'] / df['Store_Score'].mean()

        # Verify no missing values remain
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Warning: Missing values remain in columns: {missing_cols}")
            print("Filling remaining missing values with appropriate defaults...")
            df = df.fillna(df.mean())

        print("Feature engineering completed successfully.")
        return df

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return None

if __name__ == "__main__":
    # Example usage (requires a sample DataFrame)
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
            'Item_Outlet_Sales': [3735.1, 443.4, 2097.2, 732.3, 994.7]}
    sample_df = pd.DataFrame(data)

    df_with_features = create_features(sample_df)

    if df_with_features is not None:
        print("\nDataFrame with new features:")
        print(df_with_features.head())
        print("\nNew features added:")
        new_features = ['Item_MRP_Log', 'Outlet_Years', 'Item_Visibility_Normalized', 
                       'Price_Per_Weight', 'Item_MRP_Normalized', 'Store_Score', 
                       'Store_Score_Normalized']
        print(df_with_features[new_features].describe())