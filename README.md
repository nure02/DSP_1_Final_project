# Weather Prediction Project

This project focuses on predicting weather conditions based on various environmental factors. The project is divided into two main parts: Exploratory Data Analysis (EDA) and Model Training/Testing.

## Files

1. **EDA.py**: This script performs exploratory data analysis on the dataset to understand the data distribution, identify missing values, and visualize correlations between features.

2. **pre_train_test_model.py**: This script preprocesses the data, splits it into training and testing sets, and trains two machine learning models (K-Nearest Neighbors and Logistic Regression) to predict weather conditions.

## Dataset

The dataset used in this project is stored in `./data/dataset.csv`. It contains various features related to weather conditions, such as temperature, humidity, precipitation, and wind speed.

## Exploratory Data Analysis (EDA)

The `EDA.py` script performs the following tasks:

- Loads the dataset and displays basic information such as the first few rows, column names, data types, and unique values.
- Checks for missing values and visualizes them using the `missingno` library.
- Identifies and prints unique values in categorical columns to check for inconsistencies.
- Plots histograms and boxplots for numerical columns to analyze their distributions.
- Generates a correlation heatmap to visualize relationships between numerical features.

## Model Training and Testing

The `pre_train_test_model.py` script performs the following tasks:

- Encodes categorical weather conditions using `LabelEncoder`.
- Drops irrelevant columns (`date` and `weather`) and handles outliers using the Interquartile Range (IQR) method.
- Applies square root transformation to the `precipitation` and `wind` columns to normalize their distributions.
- Splits the dataset into training and testing sets.
- Trains a K-Nearest Neighbors (KNN) classifier and a Logistic Regression model.
- Evaluates the models using accuracy and Root Mean Squared Error (RMSE).
- Saves the trained models to the `models` directory using `joblib`.

