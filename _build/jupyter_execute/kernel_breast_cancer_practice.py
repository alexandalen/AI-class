## kernel breast cancer practice

# storing and analysis
import numpy as np
import pandas as pd
#visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
#Calculation and statistic  
import random
import math
import time
# For modeling
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#For performance measures
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

import datetime
import operator
plt.style.use('seaborn')
%matplotlib inline
# hide warnings
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('./kernal_breast_cancer_data.csv')
pd.set_option('display.max_column', None)
data.head(20)

data.info()

data.describe()

data.isnull().sum()

#drop unuesd column
data.drop(columns=['Unnamed: 32'], inplace=True)
data.head()

#diagnosis 做分類
data['diagnosis'].value_counts()

#有兩種方式做二分法
data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
data.head()

data.drop(columns=['id'], inplace=True)
data.head(20)

def color_negative_red(val):
    '''
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise. 
    '''
    color='red' if val<-0.8 or val>0.8 else 'black'
    return 'color: %s' % color
data.corr().style.applymap(color_negative_red)

sns.heatmap(data.corr())

X=data.iloc[:,1:]
X

y=np.array(data['diagnosis'])
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import StandardScaler
sclar=StandardScaler()
X_train=sclar.fit_transform(X_train)
X_test=sclar.transform(X_test)

#covariance matrix
cov_matrix=np.cov(X_train.T)
eigan_vals, eigan_vecs=np.linalg.eig(cov_matrix)
print('Eigen_values: \n%s' % eigan_vals)

# Total and explained variance
tot=sum(eigan_vals)
var_exp=[(i/tot) for i in sorted(eigan_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)

plt.bar(range(1,31), var_exp, alpha=1.0, align='center',label='individual explained variance')
plt.step(range(1,31), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show

#feature transformation
eigan_pair =[(np.abs(eigan_vals[i]), eigan_vecs[:,i]) for i in range (len(eigan_vals))]
eigan_pair.sort(key=lambda k: k[0], reverse=True)


w=np.hstack((eigan_pair[0][1][:, np.newaxis],
             eigan_pair[1][1][:, np.newaxis]))
print('Matrix w:\n', w)

X_train[0:5].dot(w)

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

# create default classifier
dt= DecisionTreeClassifier()
rf= RandomForestClassifier(n_estimators=100)
etc=ExtraTreesClassifier(n_estimators=100)
knc=KNeighborsClassifier()
gbm=GradientBoostingClassifier()

#train data
dt1=dt.fit(X_train,y_train)
rf1=rf.fit(X_train,y_train)
etc1=etc.fit(X_train,y_train)
knc1=knc.fit(X_train,y_train)
gbm1=gbm.fit(X_train,y_train)

#make predictions
y_pred_dt=dt1.predict(X_test)
y_pred_rf=rf1.predict(X_test)
y_pred_etc=etc1.predict(X_test)
y_pred_knc=knc1.predict(X_test)
y_pred_gbm=gbm1.predict(X_test)

#get probability values
y_pred_dt_prob=dt1.predict_proba(X_test)
y_pred_etc_prob=etc1.predict_proba(X_test)
y_pred_gbm_prob=gbm1.predict_proba(X_test)
y_pred_knc_prob=knc1.predict_proba(X_test)
y_pred_rf_prob=rf1.predict_proba(X_test)

print("DecisionTreeClassifier: {0}".format(accuracy_score(y_test,y_pred_dt)))
print('ExtraTreesClassifier: {0}'.format(accuracy_score(y_test,y_pred_etc)))
print("RandomForestClassifier: {0}".format(accuracy_score(y_test,y_pred_rf)))
print("GradientBoostingClassifier: {0}".format(accuracy_score(y_test,y_pred_gbm)))
print("KNeighborsClassifier: {0}".format(accuracy_score(y_test,y_pred_gbm)))

# calculate Confusion Matrix
print('DecisionTreeClassifier:\n', confusion_matrix(y_test, y_pred_dt))
print('ExtraTreesClassifier: \n', confusion_matrix(y_test, y_pred_dt))
print('RandomForestClassifier: \n', confusion_matrix(y_test, y_pred_dt))
print('GradientBoostingClassifier: \n', confusion_matrix(y_test, y_pred_dt))
print('KNeighborsClassifier: \n', confusion_matrix(y_test, y_pred_dt))

# drow confusion matrix
classifier =svm.SVC(kernel='linear', C=.01).fit(X_train, y_train)
np.set_printoptions(precision=2)

title_options=[('confusion matrix, without normalization',None),
               ('Nomalized confusion matrix','true')]
for title, normalize in title_options:
    disp=plot_confusion_matrix(classifier, X_test, y_test,
                               cmap=plt.cm.Blues,
                               normalize=normalize)
    disp.ax_.set_title(title)
    print (title)
plt.show()

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1, probability=True))
y_clf = pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
y_pred_prob=pipe_svc.predict_proba(X_test)
y_pred
#confusion_matrix= confusion_matrix(y_test, y_pred)
#print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# draw ROC curve (pipeline svc)
logit_roc_auc=roc_auc_score(y_test, y_pred)
fpr, tpr, threshold =roc_curve(y_test,y_pred_prob[:,1] )
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)' %logit_roc_auc)
plt.plot([0,1],[0,1], 'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()

#計算 ROC curve
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)

#計算 AUC value
print("DecisionTreeClassifier: {0}".format(auc(fpr_dt,tpr_dt)))
print("RandomForestClassifier: {0}".format(auc(fpr_rf,tpr_rf)))
print("ExtraTreesClassifier: {0}".format(auc(fpr_etc,tpr_etc)))
print("GradientBoostingClassifier: {0}".format(auc(fpr_gbm,tpr_gbm)))
print("KNeighborsClassifier: {0}".format(auc(fpr_knc,tpr_knc)))

# Plot ROC curve now
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)

# Connect diagonals
ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

# Labels etc
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')

# Set graph limits
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.1])
# Plot each graph now
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
#ax.plot(fpr_etc, tpr_etc, label = "etc")
#ax.plot(fpr_knc, tpr_knc, label = "knc")
ax.plot(fpr_gbm, tpr_gbm, label = "gbm")

# Set legend and show plot
ax.legend(loc="lower right")
plt.show()

