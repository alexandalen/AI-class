# Boston Housing

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ds = datasets.load_boston()
print(ds.DESCR)

import pandas as pd
X=pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()

y =ds.target
X.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train) #transform 轉換

X_test = scaler.transform(X_test)

lr = LinearRegression(normalize=True)

lr.fit(X_train, y_train)

lr.intercept_

lr.coef_

y_pred = lr.predict(X_test)

lr.score(X_test, y_test)

import numpy as np
np.argsort(abs(lr.coef_))

y_pred = lr.predict(X_test[0:3])
y_pred

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_squared_error(y_test, lr.predict(X_test))

r2_score(y_test, lr.predict(X_test))

