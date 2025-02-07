import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./data/dataset.csv')

print(data.head())
print(data.tail())
print(data.columns)
print(data.info())
print(data.dtypes)
print(data.nunique())
print(data.describe())
#data quality check
print(data.isnull().sum())
msno.matrix(data)
print(plt.show())
print(data.duplicated().sum())   #check duplicated rows

#Check for inconsistencies (e.g., typos in categorical data)
categorical_cols = data.select_dtypes(include=['object']).columns
print(f"\nUnique Values in Categorical Columns:")
for col in categorical_cols:
    print(f"{col}: {data[col].unique()}")

#Assess data type mismatches
print(f"\nData Type Mismatches:")
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            pd.to_numeric(data[col])
            print(f"Column '{col}' contains numeric values stored as strings.")
        except ValueError:
            pass

numerical_cols = data.select_dtypes(include=[np.number]).columns
# Plot histograms and boxplots in a grid
fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 5 * len(numerical_cols)))
fig.suptitle("Numerical Variables Analysis", fontsize=16)

for i, col in enumerate(numerical_cols):
    # Histogram
    sns.histplot(data[col], kde=True, bins=30, ax=axes[i, 0])
    axes[i, 0].set_title(f"Distribution of {col}")
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel("Frequency")

    # Boxplot
    sns.boxplot(x=data[col], ax=axes[i, 1])
    axes[i, 1].set_title(f"Boxplot of {col}")
    axes[i, 1].set_xlabel(col)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Get the value counts of each unique value in the weather column
weather_counts = data['weather'].value_counts()

# Print the percentage of each unique value in the weather column
for weather, count in weather_counts.items():
    percent = (count / len(data)) * 100
    print(f"Percent of {weather.capitalize()}: {percent:.2f}%")



# Multivariate analysis heatmap
print("\nCorrelation heatmap...")
correlation_matrix = data[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()

