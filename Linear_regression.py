import pandas as pd

dataset = pd.read_csv('C:/Users/kain_/Downloads/Course Files/005 - Regression/01Students.csv')
df = dataset.copy()

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(X_train, y_train)

Y_predict = std_reg.predict(X_test)

slr_score = std_reg.score(X_test, y_test)
slr_coesfficient = std_reg.coef_
slr_intercept = std_reg.intercept_

#equation == y = 34.27 + 5.02*x

from sklearn.metrics import mean_squared_error
import math

slr_rmse = math.sqrt(mean_squared_error(y_test, Y_predict))

import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_predict)

plt.ylim(ymin=0)
plt.show()

