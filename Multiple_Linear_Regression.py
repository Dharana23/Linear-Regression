import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

dataset = pd.read_csv('C:/Users/kain_/Downloads/Course Files/005 - Regression/02Students.csv')
df = dataset.copy()

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

std_reg = LinearRegression()
std_reg.fit(X_train, y_train)
y_predict = std_reg.predict(X_test)
mlr_score = std_reg.score(X_test, y_test)

slr_coefficient = std_reg.coef_
slr_intercept = std_reg.intercept_

#equation == y = 1.31 + 4.67*x1 + 5.07*x2

mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))

