## Bank Customer Churn Prediction

import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import datetime
import operator
plt.style.use('seaborn')
%matplotlib inline
# hide warnings
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import * #(layers, models, optimizers,preprocessing)
from tensorflow.keras.layers import * #(Conv2D, Dense, MaxPooling2D, Flatten, Dropout)
from tensorflow.keras.models import Sequential

bank= pd.read_csv('./Churn_Modelling.csv')
bank.head(10)

bank.info()

bank.isnull().sum()

bank.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
bank.head(10)

bank.describe()

# plot missing values for each variable
#bank.isnull().sum().plot()
#columns= bank.columns
#plt.xticks(np.arange(len(columns)), columns, rotation='vertical')

bank.dtypes

# Exploratory Data analysis
labels= 'Exited', 'Retained'
sizes= [bank.Exited[bank['Exited']==1].count(), 
        bank.Exited[bank['Exited']==0].count()]
explode =(0, 0.1)
fig1, ax1 =plt.subplots(figsize=(9,7))
ax1.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Proportion of customer churned and retained', size=20)
plt.show()

sns.countplot(x='Geography', hue='Exited', data=bank).set_title('Countplot-Geography Column')
#hue 設定標示的名稱

sns.countplot(x='Gender', hue='Exited', data=bank).set_title('Countplot-Gender Column')

sns.countplot(x='HasCrCard', hue = 'Exited',data = bank).set_title('Countplot-HasCreditCard Column')

sns.countplot(x='IsActiveMember', hue = 'Exited',data = bank).set_title('Countplot-IsActiveMember Column')

bank.corr()

#用不同顏色來定義 high correlation 和 low correlation
def color_negatve_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color= 'red' if val<-0.8 or val>0.8 else 'black'
    return 'color: %s' % color
bank.corr().style.applymap(color_negatve_red)

#seaborn作圖
sns.heatmap(bank.corr())

#基於連續變數 製作盒鬚圖
fig, axarr = plt.subplots(3,2, figsize=(20,12))
sns.boxplot(y='CreditScore', x='Exited', 
            hue ='Exited', data= bank, 
            ax=axarr[0][0]).set_title('Boxplot-Credit Score Column')
sns.boxplot(y='Age', x='Exited', 
            hue ='Exited', data= bank, 
            ax=axarr[0][1]).set_title('Boxplot-Age Column')
sns.boxplot(y='Tenure', x='Exited', 
            hue ='Exited', data= bank, 
            ax=axarr[1][0])
sns.boxplot(y='Balance', x='Exited', 
            hue ='Exited', data= bank, 
            ax=axarr[1][1])
sns.boxplot(y='NumOfProducts', x='Exited', 
            hue ='Exited', data= bank, 
            ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary', x='Exited', 
            hue ='Exited', data= bank, 
            ax=axarr[2][1])




# Feature Engineering
bank['BalanceSalaryRatio']=bank.Balance/bank.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio', x= 'Exited', hue='Exited',
            data=bank)
plt.ylim(-1,5) #設定y軸數值(min,max)

bank['TenureByAge']=bank.Tenure/(bank.Age)
sns.boxplot(y='TenureByAge', x= 'Exited', hue='Exited',
            data=bank)
plt.ylim(-0.2, 0.7)
plt.show()

bank['CreditScoreGivenAge'] = bank.CreditScore/(bank.Age)
sns.boxplot(y='CreditScoreGivenAge',x = 'Exited', hue = 'Exited',
            data = bank)
plt.show()

#Data Preparation for the Model fitting

#連續變量
continuous_vars=['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts',
                 'EstimatedSalary', 'BalanceSalaryRatio','TenureByAge','CreditScoreGivenAge']
#絕對變量
categorical_vars=['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df=bank[['Exited']+continuous_vars+categorical_vars]
df.head()

sns.set()
sns.set(font_scale = 1.25)
sns.heatmap(df[continuous_vars].corr(), annot = True,fmt = ".1f")
plt.show()

'''
Changing values of column HasCrCard and IsActiveMember from 0 to -1 
so that they will influence negatively to the model instead of no effect.
'''
df.loc[df.HasCrCard == 0, 'HasCrCard'] = -1
df.loc[df.IsActiveMember == 0, 'IsActiveMember'] = -1
df.head()

# 建立One-hot encoding categorical columns
print (df['Gender'].unique())
print (df['Geography'].unique())

from sklearn.preprocessing import LabelEncoder 
label=LabelEncoder()
df['Gender']=label.fit_transform(df['Gender'])
df['Geography']=label.fit_transform(df['Geography'])
df.head()

df1 = pd.get_dummies(data=df, columns=['Gender','Geography'])
df1.columns


df1.head()

continuous_vars

#標準化 (使用minmaxsclar)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df1[continuous_vars]=scaler.fit_transform(df1[continuous_vars])

for col in df1:
    print(f'{col}: {df1[col].unique()}')

# Model fitting and selection

# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#分群 training
X= df1.drop('Exited', axis='columns')
y= df1['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Logistic Regression
logreg= LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log= round(logreg.score(X_test, y_test)*100,2)
acc_log

#準確率
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred)*100))

# SVM
svc =SVC()
svc.fit(X_train, y_train)
y_pred1=svc.predict(X_test)
acc_svc=round(svc.score(X_test,y_test)*100,2)
acc_svc

#準確率
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred1)*100))

# knn
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred2=knn.predict(X_test)
acc_knn= round(knn.score(X_test,y_test)*100,2)
acc_knn

#準確率
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred2)*100))

# Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred3 = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred3)*100))

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred4 = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred4)*100))

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred5 = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred5)*100))

#XGBoost ( Extreme Gradient Boosting )
XGB= XGBClassifier(base_score=0.5,  booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=5,
                    min_child_weight=1, missing=None, n_estimators=100,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, subsample=1)
XGB.fit(X_train,y_train)
y_pred6 = XGB.predict(X_test)
acc_XGB= round(XGB.score(X_test,y_test)*100, 2)
print('accuracy={:.2f}\n'.format(accuracy_score(y_test, y_pred6)*100))

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Decision Tree', 'XGBoots'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_decision_tree, acc_XGB]})
models.sort_values(by='Score', ascending=False)

from sklearn.metrics import classification_report
'''
print('\033[1m'+"輸入文字") 粗體字
print('\033[1;34m'____文字_____) 34m 指顏色代碼
'''
print("-"*60)
print('\033[1m' + '\033[1;34m Logistic Regression \033[0m:')
print(classification_report(y_test, y_pred))
print("-"*60)
print('\033[1m' + '\033[1;34m SVM \033[0m:') 
print(classification_report(y_test, y_pred1))
print("-"*60)
print('\033[1m' + '\033[1;34m KNN \033[0m:') 
print(classification_report(y_test, y_pred2))
print("-"*60)
print('\033[1m' + '\033[1;34m Naive Bayes \033[0m:') 
print(classification_report(y_test, y_pred3))
print("-"*60)
print('\033[1m' + '\033[1;34m Decision Tree \033[0m:') 
print(classification_report(y_test, y_pred4))
print("-"*60)
print('\033[1m' + '\033[1;34m Random Forest \033[0m:') 
print(classification_report(y_test, y_pred5))
print("-"*60)
print('\033[1m' + '\033[1;34m XGBoost ( Extreme Gradient Boosting ) \033[0m:') 
print(classification_report(y_test, y_pred6))

#Using Artificial Neural Network technique
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

model= Sequential()

# first hidden layer
model.add(Dense(8, activation='relu', input_dim=16))
model.add(Dropout(0.1))

# second hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))

# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the NN
# binary_crossentropy loss function used when a binary output is expected
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
train_history=model.fit(X_train, y_train, batch_size = 10, epochs = 50)

plt.plot(train_history.history['loss'],'r')
plt.plot(train_history.history['accuracy'],'g')

# Evaluating test data with this model and accuracy is coming to be 85.85 %
model.evaluate(X_test, y_test)

X_test[:10]

# Manually verifying some predictions
yp = model.predict(X_test)
yp[:10]

y_predict=[]
for element in yp:
    if element > 0.5:
        y_predict.append(1)
    else:
        y_predict.append(0)
# element>0.5 會編列成1 反之, 編列成0
y_predict[:10]

y_test[:10]

print(classification_report(y_test, y_predict))

# confusion matrix
cm=tf.math.confusion_matrix(labels=y_test, predictions=y_predict)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
#fmt:字串格式程式碼，矩陣上標識數字的資料格式，比如保留小數點後幾位數字
plt.xlabel('Predicted')
plt.ylabel('Truth')

