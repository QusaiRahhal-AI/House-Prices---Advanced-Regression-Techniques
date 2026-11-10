# House Prices - Advanced Regression Techniques

## Project Overview
An end-to-end regression pipeline to predict residential property sale prices based on various structural, location, and qualitative features. The focus of this project is advanced feature engineering, outlier detection, and strict regularization to mitigate overfitting.

## Dataset
- **Source:** Kaggle - House Prices: Advanced Regression Techniques
- **Dataset Link:** [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data]

## Pipeline Architecture & Methodology
1. **Data Preprocessing & Cleaning:**
   - Addressed missing values across high-dimensional numerical and categorical features.
   - Handled data types and applied appropriate encoding strategies.
2. **Outlier Detection & Handling:**
   - Identified and managed numerical anomalies using the Interquartile Range (IQR) method to prevent distortion in regression models.
3. **Model Training & Regularization:**
   - Trained linear and tree-based models including Ridge Regression, Random Forest, and XGBoost.
   - Applied hyperparameter tuning (controlling tree depth, learning rate, and regularization parameters) to resolve significant overfitting gaps between training and testing sets.

## Results & Performance
- **Best Performing Model:** Tuned XGBoost and Ridge Regression
- **Key Metrics:** Achieved stable high R-squared ($R^2$) scores on the test set, demonstrating robust generalization and minimal variance between train and test performance.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
