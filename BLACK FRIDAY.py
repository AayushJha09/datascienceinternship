import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('C:/Users/admi/Downloads/Black Friday Sales/train.csv')
test_df = pd.read_csv('C:/Users/admi/Downloads/Black Friday Sales/test.csv')

print('Missing Values:')
print(train_df.isnull().sum())

print("Train Dataset Columns:", train_df.columns)
print("Test Dataset Columns:", test_df.columns)

combined_df = pd.concat([train_df, test_df])
print(combined_df)

le = LabelEncoder()
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
for col in categorical_cols:
    combined_df[col] = le.fit_transform(combined_df[col])

train_df = combined_df[:len(train_df)]

test_df = combined_df[len(train_df):]

X_train = train_df.drop('Purchase', axis=1)
X_train = X_train.drop('Product_Category_2', axis=1)
X_train = X_train.drop('Product_Category_3', axis=1)
X_train = X_train.drop('Product_ID', axis=1)

y_train = train_df['Purchase']

#testing the dataset using linear regression model
lreg = LinearRegression()
lreg.fit(X_train,y_train)

X_test = test_df.drop('Purchase', axis=1)
X_test = X_test.drop('Product_Category_2', axis=1)
X_test = X_test.drop('Product_Category_3', axis=1)
X_test = X_test.drop('Product_ID', axis=1)
y_pred = lreg.predict(X_test)

# Print the predicted purchase amounts
print("Predicted Purchase Amounts:")
print(y_pred)