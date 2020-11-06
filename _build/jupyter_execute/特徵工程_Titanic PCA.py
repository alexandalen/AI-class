# Titanic PCA and 分布圖

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 

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

df.drop(['parch','who','alone','embarked', 'alive', 'class','adult_male'], axis=1, inplace=True)
df.head()

df.info()

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

#covariance matrix
cov_mat = np.cov(X_train.T)
eigen_vals, eigen_vecs= np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

#Total and explained variance
tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 7), var_exp, alpha=1.0, align='center', label='individual explained variance')
plt.step(range(1, 7), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show


#Feature transformation
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w =np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W:\n', w)

X_train[0].dot(w)

X_train_pca=X_train.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

## sklearn PCA

pca= PCA()
X_train_pca=pca.fit_transform(X_train)
pca.explained_variance_ratio_

plt.bar(range(1, 7), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 7), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()

pca=PCA(n_components=2)
X_train_P=pca.fit_transform(X_train)
X_test_P=pca.transform(X_test)

plt.scatter(X_train_P[:, 0], X_train_P[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

