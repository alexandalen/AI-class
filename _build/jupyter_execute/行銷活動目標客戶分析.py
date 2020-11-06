## 行銷活動目標客戶分析

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

df=pd.read_csv('./banking.csv')
data=pd.DataFrame(df)
#print (data)
print(data.shape)

data.head()

data.isnull().sum()

data.isnull().sum().sum()

data['education'].unique()

data['education'].value_counts()

#將 basic 4y,6y,9y 群組 統稱 basic
data['education']=np.where(data['education']=='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education']=='basic 6y', 'Basic', data['education'])
data['education']=np.where(data['education']=='basic 4y', 'Basic', data['education'])

data['education'].unique()

# Data exploration
#y=1 (yes) y=0(no)
data['y'].value_counts()

sns.countplot(x='y', data=data, palette='hls')
plt.show()

No_sub=len(data[data['y']==0])
yes_sub=len(data[data['y']==1])
percent_No_sub=No_sub/(yes_sub+No_sub)*100
print ('percent of no subscription:', round(percent_No_sub,2))
percent_yes_sub=yes_sub/(yes_sub+No_sub)*100
print ('percent of yes subscription:', round(percent_yes_sub,2))

data.groupby('y').mean()

data.groupby('job').mean()

data.groupby('marital').mean()

data.groupby('education').mean()

# Visualization
%matplotlib inline
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('purchase frequency for Job')
plt.xlabel('job')
plt.ylabel('Frequency of purchase')

pd.crosstab(data.job, columns=[data.month, data.y])

table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Material status vs Purchase')
plt.xlabel('Material status')
plt.ylabel('Proportion of Custmers')

#education seems a good predictor of the outcome variable
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')

# day of week may not be a good predictor
pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('purchase frequency for day of week')
plt.xlabel('Day of week')
plt.ylabel('Frequency of purchase')

# month might be a good predictor
pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')

data.age.hist()
plt.title('Histogram of age')
plt.xlabel('Age')
plt.ylabel('Frequency')

pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')

# create dummy variables
cat_vars=['job', 'marital', 'education','default', 'housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list=pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
    
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final.columns.values

len(data_final.columns)

# over-sampling using SMOTE
X=data_final.loc[:, data_final.columns !='y']
y=data_final.loc[:, data_final.columns == 'y']

from imblearn.over_sampling import SMOTE

os=SMOTE(random_state=0)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.3, random_state=0)
columns = X_train.columns

os_data_X, os_data_y= os.fit_sample(X_train, y_train)
os_data_X= pd.DataFrame(data=os_data_X, columns=columns)
os_data_y=pd.DataFrame(data=os_data_y, columns=['y'])
#can check the numbers of our data
print('length of oversampled data is', len(os_data_X))
print('Number of no subscription in oversampled data', len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

# Recursive feature elimination
data_final_vars= data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg =LogisticRegression()

rfe=RFE(logreg, 20)
rfe=rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']

# implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result= logit_model.fit()
print(result.summary2())

## The p-values for one variable is very high, therefore, we will remove them.

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
X=os_data_X[cols]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result= logit_model.fit()
print(result.summary2())

## Logistic Regression Model Fitting

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.3, random_state=0)
logreg= LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

## Confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

## ROC curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc=roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, threshold = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
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

