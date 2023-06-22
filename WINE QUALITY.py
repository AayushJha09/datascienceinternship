import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state = 0)
gbn = GradientBoostingClassifier(n_estimators = 10)
rf = RandomForestClassifier(random_state=3)


df = pd.read_csv("C:/Users/admi/Downloads/Wine Quality/winequalityN.csv")
print(df)
df.head()
print(df)
print(df.isnull().sum())

df['fixed acidity'].fillna((df['fixed acidity']).mean(), inplace = True)
df['pH'].fillna((df['pH']).mean(), inplace = True)

print(df.isnull().sum())
X = df.drop('fixed acidity', axis = 1)
X = X.drop('volatile acidity', axis =1)
X = X.drop('citric acid', axis =1)
X = X.drop('residual sugar', axis =1)
X = X.drop('chlorides', axis =1)
X = X.drop('free sulfur dioxide', axis =1)
X = X.drop('total sulfur dioxide', axis =1)
X = X.drop('density', axis =1)
X = X.drop('sulphates', axis =1)

Y = df['quality']
print(X.isnull().sum())

df['pH'].fillna((df['pH']).mean(), inplace = True)
print(X.isnull().sum())

le = LabelEncoder()
le.fit(X['type'])
(X['type'])=le.transform(X['type'])
print((X['type']))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=3,test_size = 0.2)

#Random Forest Classifier
#rf.fit(X_train, Y_train)

#Logistic Regression
lr.fit(X_train, Y_train)
y_pred=lr.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy :", accuracy)
