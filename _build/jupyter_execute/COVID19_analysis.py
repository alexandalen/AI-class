## COVID-19 (mortality and recovery analysis and EDA)

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
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime
import operator
plt.style.use('seaborn')
%matplotlib inline
# hide warnings
import warnings
warnings.filterwarnings('ignore')

confirm_df= pd.read_csv('./time_series_covid19_confirmed_global.csv')
death_df = pd.read_csv('./time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('./time_series_covid19_recovered_global.csv')

pd.set_option('display.max_column', None)
confirm_df.head(20)

cols=confirm_df.keys()

confirmed = confirm_df.loc[:, cols[4]:cols[-2]]
deaths = death_df.loc[:, cols[4]:cols[-2]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-2]]

dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 
china_cases = [] 
italy_cases = []
us_cases = [] 

for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovery_sum=recoveries[i].sum()
    
    #confirmed(確診), deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovery_sum)
    total_active.append(confirmed_sum-death_sum-recovery_sum)
    
    #計算rates
    #致死率和回復率
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovery_sum/confirmed_sum)
    
    # case studies (個案研究)
    china_cases.append(confirm_df[confirm_df['Country/Region']=='China'][i].sum())
    italy_cases.append(confirm_df[confirm_df['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirm_df[confirm_df['Country/Region']=='US'][i].sum())

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered= np.array(total_recovered).reshape(-1,1)

# Future forcasting(未來預測)
day_in_future=10
future_forcast = np.array([i for i in range(len(dates)+day_in_future)]).reshape(-1,1)
adjusted_dates= future_forcast[:-10]

#將整數轉換為日期時間以獲得更好的視覺化
start='1/22/2020'
start_date= datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates=[]
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

#做資料切割(condirmed)
X_train, X_test, y_train, y_test=train_test_split(days_since_1_22, world_cases, test_size=.2, shuffle=False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#製作預測確診個案模型，並使用support vector machine, bayesian ridge , and linear regression
#use this to find the optimal parameters for SVR
#c = [0.01, 0.1, 1]
#gamma = [0.01, 0.1, 1]
#epsilon = [0.01, 0.1, 1]
#shrinking = [True, False]
#degree = [3, 4, 5]

#svm_grid = {'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking, 'degree': degree}

#svm = SVR(kernel='poly')
#svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
#svm_search.fit(X_train, y_train)
#svm_search.best_params_

svm_confirmed= SVR(shrinking=True, kernel='poly', gamma=.01, epsilon=1, degree=6, C=.1)
svm_confirmed.fit(X_train, y_train)
svm_pred=svm_confirmed.predict(future_forcast)

# 檢查測試數據，看MAE 跟 MSE
svm_test_pred=svm_confirmed.predict(X_test)
plt.plot(svm_test_pred)
plt.plot(y_test)
print('MAE value:', mean_absolute_error(svm_test_pred, y_test))
print('MSE value:', mean_squared_error(svm_test_pred, y_test))

#轉換data 來進行polynomial regression
poly= PolynomialFeatures(degree=5)
poly_X_train=poly.fit_transform(X_train)
poly_X_test=poly.fit_transform(X_test)
poly_future_forcast= poly.fit_transform(future_forcast)

#polynomial regression
linear_model= LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train,y_train)
test_linear_pred=linear_model.predict(poly_X_test)
linear_pred=linear_model.predict(poly_future_forcast)
print('MAE value:', mean_absolute_error(test_linear_pred, y_test))
print('MSE value:', mean_squared_error(test_linear_pred, y_test))
print(linear_model.coef_)

plt.plot(test_linear_pred)
plt.plot(y_test)

#貝氏分析 (bayesian ridge regression)
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}
#bayesian ridge polynomial regression
bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', 
                                     cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train, y_train)

bayesian_search.best_params_

bayesian_confirmed= bayesian_search.best_estimator_
test_bayesian_pred= bayesian_confirmed.predict(poly_X_test)
bayesian_pred= bayesian_confirmed.predict(poly_future_forcast)
print('MAE value:', mean_absolute_error(test_bayesian_pred, y_test))
print('MSE value:', mean_squared_error(test_bayesian_pred, y_test))

plt.plot(y_test)
plt.plot(test_bayesian_pred)

#做出確診病例數，活動病例數，死亡，康復，死亡率和康復率的圖表
'''
label: 標籤文字
fontdict: 標籤文字字型字典，例如字型系列、顏色、粗細和大小
labelpad: 標籤和 x or y 軸之間的間距
linewidth 或 lw: 線寬，以 points 為單位
'''
plt.figure(figsize=(16,9))
plt.plot(adjusted_dates, world_cases, lw=4)
plt.title('number of COVID-19 cases overtime ', size=30)
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Number of cases', labelpad = 20, size=24)
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

#COVID-19 case confirmed in China, Italy, US
plt.figure(figsize=(16,9))
plt.plot(adjusted_dates, china_cases, lw=4)
plt.plot(adjusted_dates, italy_cases, lw=4)
plt.plot(adjusted_dates, us_cases, lw=4)
plt.title('Confirmed number of COVID-19 cases', size=30)
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Confirmed number of cases', labelpad = 20, size=24)
plt.legend(['China', 'Italy', 'US'], prop={'size': 20})
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

#SVM preidction
plt.figure(figsize=(16,9))
plt.plot(adjusted_dates, world_cases,lw=4)
plt.plot(future_forcast,svm_pred, linestyle='--', color='purple', lw=4)
plt.title('Confirmed number of COVID-19 cases overtime', size=30)
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Confirmed number of cases', labelpad = 20, size=24)
plt.legend(['Confirmed cases', 'SVM prediction'], prop={'size': 20})
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

#active death and recovary case
plt.figure(figsize=(16,9))
plt.plot(adjusted_dates, total_active, color='purple', lw=4)
plt.title('number of COVID-19 active cases overtime ', size=30)
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Number of active cases', labelpad = 20, size=24)
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

plt.figure(figsize=(16,9))
plt.plot(adjusted_dates,total_deaths, linestyle='--', color='red', lw=4)
plt.title('number of COVID-19 death cases overtime ', size=30)
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Number of death cases', labelpad = 20, size=24)
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

plt.figure(figsize=(16,9))
plt.plot(adjusted_dates,total_recovered, linestyle='-.', color='green', lw=4)
plt.title('number of COVID-19 recovery cases overtime ', size=30)
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Number of recovery cases', labelpad = 20, size=24)
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

mean_mortality_rate = round(np.mean(mortality_rate),5)
plt.figure(figsize=(16,9))
plt.plot(adjusted_dates, mortality_rate, color='yellow', lw=4)
plt.axhline(y=mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality rate of COVID-19 overtime ', size=30)
plt.legend(['Mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Mortality rate', labelpad = 20, size=24)
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

mean_recovery_rate = round(np.mean(recovery_rate),5)
plt.figure(figsize=(16,9))
plt.plot(adjusted_dates, recovery_rate, color='blue', lw=4)
plt.axhline(y=mean_recovery_rate, linestyle='--', color='black')
plt.title('Recovery rate of COVID-19 overtime ', size=30)
plt.legend(['Recovery rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})
plt.xlabel('Days since 1/22/2020', labelpad = 20, size=24)
plt.ylabel('Recovery rate', labelpad = 20, size=24)
plt.xticks(size=18, fontweight = "bold")
plt.yticks(size=18, fontweight = "bold")
plt.show()

# Exploratory Data Analysis(EDA) 計算
confirmed_df=pd.read_csv('./new.csv')
pd.set_option('display.max_column', None)
confirmed_df.head()

death_df.head()

df1=confirmed_df.drop('Province/State', axis=1)

'''
使用groupby()方法可以將資料依照自己要的column分組
，Country/Region用的內容做分組的依據，並存到變數內
DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
ascending这个参数的默认值是True，按照升序排序，当传入False时，按照降序进行排列
'''
#因為 death data 統計到 3/26/20 所以要把confirm_df 做群組
df2=df1.groupby(by='Country/Region').sum().sort_values('3/26/20', ascending=False)
df2.head(20)

#選前20個並delete 'lat', 'long'
top_N=20
df3=df2.iloc[:top_N+1]
for col in ['Lat','Long']:
    df3.drop(col, inplace=True, axis=1)
df3=df3[df3.index !='China'] # != 相當於 ==
df3

data_list=[]
country_list=df3.index
first_day=[0 for _ in range(top_N)]

for i in range(top_N):
    #取每個國家第一天>0
    for col in range(0, len(df3.columns)):
        if df3.iloc[i, col]>50:
            first_day[i]=col
            break
            
for i in range(top_N):
    data_list.append([])
    for col in range(first_day[i], len(df3.columns)):
        data_list[i].append(df3.iloc[i,col])
        
df=pd.DataFrame(data_list)
df.index=country_list
df

#給予圖表註解 並避開 nan
def set_annotate(i):
    if len(df.columns)>60 and not np.isnan(df.iloc[i,60]):
        plt.annotate(df.index[i], (60, df.iloc[i,60]))
    elif len(df.columns)>50 and not np.isnan(df.iloc[i,55]):
        plt.annotate(df.index[i], (55, df.iloc[i,55]))
    elif len(df.columns)>50 and not np.isnan(df.iloc[i,50]):
        plt.annotate(df.index[i], (50, df.iloc[i,50]))

#製圖
plt.figure(figsize=(16,12))
for i in range(5):
    plt.plot(df.iloc[i],lw=3)
    #set annotate
plt.legend(df.index)
plt.title('EDA of COVID-19 cases overtime ', size=30)
plt.xlabel('Days of confirm number over 50 ', labelpad = 20, size=24)
plt.ylabel('Number of confirm number', labelpad = 20, size=24)
plt.xticks(size=14, fontweight = "bold")
plt.yticks(size=14, fontweight = "bold")
plt.show()

