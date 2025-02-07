import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error

# Load dataset
data = pd.read_csv('./data/dataset.csv')

# Create a label encoder object
le = LabelEncoder()

# Fit the encoder to the weather column and transform the values
data['weather_encoded'] = le.fit_transform(data['weather'])

# Create a dictionary that maps the encoded values to the actual names
weather_names = dict(zip(le.classes_, le.transform(le.classes_)))

# Drop the "date" column from the dataframe
data = data.drop("date", axis=1)
data = data.drop("weather", axis=1)


# Calculate the first quartile (Q1), third quartile (Q3), and interquartile range (IQR)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Align the indices of df and (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)
data, _ = data.align((Q1 - 1.5 * IQR) | (data> (Q3 + 1.5 * IQR)), axis=1, copy=False)

# Remove outliers using the IQR method
data = data.dropna()

# Take the square root of the "precipitation" column
data["precipitation"] = np.sqrt(data["precipitation"])

# Take the square root of the "wind" column
data["wind"] = np.sqrt(data["wind"])
print(data.head())

#split the dataset into train and test set
x = ((data.loc[:,data.columns!="weather_encoded"]).astype(int)).values[:,0:]
y = data["weather_encoded"].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)

#Model traaining
from sklearn.neighbors import KNeighborsClassifier

# create a KNN classifier and fit it to the training data
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# calculate the accuracy score of the KNN classifier on the test data
knn_score = knn.score(x_test, y_test)
print("KNN Accuracy:", knn_score)

y_pred = knn.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)

print(f'RMSE in KNN: {rmse}')

# Import the LogisticRegression class from Scikit-learn
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression classifier
lg = LogisticRegression()

# Train the Logistic Regression classifier on the training data
lg.fit(x_train, y_train)

# Calculate the accuracy of the Logistic Regression classifier on the test data
lg_score = lg.score(x_test, y_test)

# Print the accuracy of the Logistic Regression classifier
print(f"Logistic Regression Accuracy: {lg_score}")

# evaluate the model

y_pred = lg.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)

print(f'RMSE in logistic regression: {rmse}')

# save the model
import os
import joblib

os.makedirs('models', exist_ok=True)

joblib.dump(lg, 'models/lg.pkl')
joblib.dump(knn, 'models/knn.pkl')

