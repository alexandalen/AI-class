# classfication_Wine

from sklearn.datasets import load_wine
ds = load_wine()

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

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn.score(X_test, y_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

svc.score(X_test, y_test)

from sklearn.metrics import confusion_matrix

y_pred=knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))

