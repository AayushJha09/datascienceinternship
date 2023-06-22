import pandas as pd
import numpy as np
features = pd.read_csv("C:Users/admi/Downloads/walmart-recruiting-store-sales-forecasting/features.csv/features.csv")
train = pd.read_csv("C:/Users/admi/Downloads/walmart-recruiting-store-sales-forecasting/train.csv/train.csv")
stores = pd.read_csv("C:/Users/admi/Downloads/walmart-recruiting-store-sales-forecasting/stores.csv")
test1 = pd.read_csv("C:/Users/admi/Downloads/walmart-recruiting-store-sales-forecasting/test.csv/test.csv")

df1 = train.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')
df = df1.merge(stores, on = ['Store'], how = 'inner')
df.head()

print(df.isnull().sum())

df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1, inplace = True)


df['Date'] = pd.to_datetime(df['Date'])
df.set_index(keys = "Date", inplace = True)

columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']

Q3 = df[columns].quantile(.75)
Q1 = df[columns].quantile(.25)
IQR = Q3 - Q1
UL = Q3 + 1.5*IQR
LL = Q1 -  1.5*IQR

for column in columns:
    df[column] = np.where(df[column] > UL[column], UL[column], np.where(df[column] < LL[column], LL[column], df[column]))

df_test = test1.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')
test = df_test.merge(stores, on = ['Store'], how = 'inner')

test.drop(axis = 1, columns = ["MarkDown1", "MarkDown2","MarkDown3","MarkDown4", "MarkDown5"], inplace = True)

print(test.isnull().sum())
test['CPI'] = test['CPI'].fillna(test['CPI'].mean())
test['Unemployment'] = test['Unemployment'].fillna(test['Unemployment'].mean())

columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']

Q3 = test[columns].quantile(.75)
Q1 = test[columns].quantile(.25)
IQR = Q3 - Q1
UL = Q3 + 1.5*IQR
LL = Q1 -  1.5*IQR

for column in columns:
    test[column] = np.where(test[column] > UL[column], UL[column], np.where(test[column] < LL[column], LL[column], test[column]))

test['Date'] = pd.to_datetime(test['Date'])
test.set_index(keys = 'Date', inplace = True)
df_test.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['IsHoliday'] = encoder.fit_transform(df['IsHoliday'])
df['Type'] = encoder.fit_transform(df['Type'])
test['IsHoliday'] = encoder.fit_transform(test['IsHoliday'])
test['Type'] = encoder.fit_transform(test['Type'])
df['CPI'] = df['CPI'].round(2)

x = df.drop(['Weekly_Sales'], axis = 1)
y = df['Weekly_Sales']

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 10)

print("x Train Shape :",x_train.shape)
print("x Val Shape   :",x_val.shape)
print("y Train Shape :",y_train.shape)
print("y Val Shape   :",y_val.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_val)
lr.score(x_val, y_val)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_pred, y_val)
r2 = r2_score(y_pred, y_val)
print('Mean Square Error = ', mse)
print('R2 Score = ', r2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt_model = dt.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_val)
rms_dt = np.sqrt(mean_squared_error(y_pred_dt, y_val))
r2_dt = r2_score(y_pred_dt, y_val)
print('RMSE of DT = ', rms_dt)
print('R2 Score of DT = ', r2_dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_model = rf.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_val)

rms_rf = np.sqrt(mean_squared_error(y_pred_rf, y_val))
r2_rf = r2_score(y_pred_rf, y_val)
print('RMSE of RF = ', rms_rf)
print('R2 Score of RF = ', r2_rf)