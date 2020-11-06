# Boston 房價預測

from sklearn.datasets import load_boston
ds = load_boston()

print(ds.DESCR)

import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()

y = ds.target
y

X.isnull().sum()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test[0]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr.coef_

lr.intercept_

from sklearn.metrics import mean_squared_error

y_pred = lr.predict(X_test)
mean_squared_error(y_test, y_pred)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)
mean_squared_error(y_test, y_pred)

r2_score(y_test, y_pred)

