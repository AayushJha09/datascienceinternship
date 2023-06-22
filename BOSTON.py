import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/admi/Downloads/housing.csv")
#print(df.keys())
# print(df.target)
# print(df.target_names)
# print(df.head())
# print(df.describe())
# print(df.isnull().sum())
# print(df.columns)

# data = pd.DataFrame(df.data, columns=boston.feature_names)
# data['Price'] = df.target
# X = data.drop('PRICE', axis=1)
# y = data['PRICE']
Y = df['MEDV']
X = df.drop('MEDV', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=42)

linR = LinearRegression()
linR.fit(X_train, Y_train)

y_pred = linR.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Mean Squared Error", mse)
print("R-squared", r2)