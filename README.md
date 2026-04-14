# E-Commerce Furniture Sales Prediction & Market Analysis

**Author:** Ashutosh Ranjan  
**Program:** M.Tech Data Science, JNU  

## Project Overview
This project analyzes an E-commerce Furniture Dataset (2024) containing 2,000 scraped listings. The objective is to build a machine learning pipeline that predicts the number of items sold (`sold`) based on product attributes, pricing, and promotional tags.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** VS Code / Jupyter

## Pipeline Details
1. **Data Preprocessing:** Cleaned unstructured pricing data and handled missing values. Encoded categorical shipping tags using `LabelEncoder`.
2. **Feature Engineering:** Converted unstructured `productTitle` text into numerical features using `TfidfVectorizer` to extract top keywords.
3. **Exploratory Data Analysis (EDA):** Generated visual distributions of price and its non-linear impact on units sold.
4. **Modeling:** Trained both a Linear Regression model and a Random Forest Regressor to predict sales volume.

## Model Performance
* **Linear Regression:** R-Squared = 0.02 | MSE = 25923.96
* **Random Forest Regressor:** R-Squared = 0.72 | MSE = 7498.35

## Conclusion
The Random Forest model significantly outperformed Linear Regression, capturing 72% of the variance in sales. This indicates that customer purchasing behavior is highly non-linear and relies heavily on a combination of specific product keywords, shipping tags, and price points rather than just the price alone.