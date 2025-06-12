# BigMart Sales Prediction

## Overview
This Python-based project predicts sales for products in BigMart stores using historical data and machine learning techniques. It utilizes libraries like pandas and scikit-learn for data processing and model development, providing insights for better inventory management.

## Project Structure
```
├── src/
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── model/
│   │   ├── sales_predictor.py
│   │   └── data_preprocessor.py
│   └── main.py
├── data/
│   └── bigmart.csv
├── requirements.txt
├── README.md
└── project_proposal.md
```

## Prerequisites
- Python 3.x
- Libraries: pandas, scikit-learn, numpy

## Setup Instructions
1. Clone the repository
2. Place the BigMart dataset (bigmart.csv) in the `data` directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```bash
   python src/main.py
   ```

## Features
- Comprehensive data preprocessing and cleaning
- Advanced feature engineering including:
  - Missing value handling for Item_Weight and Outlet_Size
  - Normalized item visibility and MRP
  - Price per weight calculations
  - Store performance indicators
- Multiple regression models (Linear, Ridge, Lasso, Random Forest)
- Extensive model evaluation metrics:
  - RMSE: 1062.18
  - MAE: 741.72
  - R2 Score: 0.585
  - MAPE: 55.22%
  - Cross-validation RMSE: 1146.73 (±37.54)

## Testing
The model has been thoroughly tested with cross-validation and multiple evaluation metrics. To test new predictions:

1. Ensure your data follows the same format as the training dataset
2. Place your test data in the `data` directory
3. Run the main script as described in the Usage section
---
