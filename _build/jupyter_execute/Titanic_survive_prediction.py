# Titanic survive prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= sns.load_dataset("titanic")
df.head(10)

df['survived'].value_counts()
df['survived'].unique()
df.corr()

df["age"].fillna(df['age'].median(),inplace=True)

df[pd.isna(df['embark_town'])]
df.iloc[[61-1,829-1]]

df['embark_town'].fillna(method='ffill', inplace=True)
df.iloc[[61,829]]

df.isnull().sum()

df["sex"]= df['sex'].map({'male':1, 'female':0})
df['embark_town']=df['embark_town'].map({"Southampton":1, "Cherbourg":2, "Queenstown":3})
df["embarked"]= df["embarked"].map({'S':1, 'C':2, 'Q':3})
df["class"]= df['class'].map({'First':1, 'Second':2, 'Third':3})
df["who"]= df['who'].map({'man':1, 'woman':2, 'child':3})
df["adult_male"]= df['adult_male'].map({True:1, False:0})
df["alive"]= df['alive'].map({'yes':1, 'no':0})
df["alone"]= df['alone'].map({True:1, False:0})
df.head()

df.drop('deck', axis=1, inplace=True)
df.head()

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < -0.8  or val>0.8 else 'black'
    return 'color: %s' % color
df.corr().style.applymap(color_negative_red)

sns.heatmap(df.corr())

df.drop(['who','alone','embarked', 'alive', 'class','adult_male'], axis=1, inplace=True)
df.head()

bins=[0, 15, 35, 60, 100]
pd.cut(df['age'], bins, labels=[0,3,2,1]) 
df['age']=pd.cut(df['age'], bins, labels=[0,3,2,1]) 
print(pd.value_counts(df['age']).sort_index)

X = df.iloc[:, 1:]
X

y=np.array(df['survived'])
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2)

from sklearn.preprocessing import StandardScaler
sclar=StandardScaler()
X_train=sclar.fit_transform(X_train)
X_test=sclar.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
log_pre = round(clf.score(X_test,y_test)*100, 2)
log_pre

from sklearn.ensemble import RandomForestClassifier
clf2=RandomForestClassifier(n_estimators=300)
clf2.fit(X_train, y_train)
Y_pred = clf2.predict(X_test)
acc_random_forest = round(clf2.score(X_test, y_test) * 100, 2)
acc_random_forest

from joblib import dump, load
dump(sclar, 'std1.joblib')

dump(clf2, 'model.joblib')

