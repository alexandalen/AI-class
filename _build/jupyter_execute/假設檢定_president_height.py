# 統計 (t-test)

import pandas as pd
from scipy import stats
import numpy as np  
df= pd.read_csv('./president_heights.csv')
d=pd.DataFrame(df)
print(d[['height(cm)']])



last=d.tail(10)
print(last[['height(cm)']])


early=df.head(len(df)-10) 
print(early[['height(cm)']])

s1=last[['height(cm)']]
s2=early[['height(cm)']]

#sample=np.append(s1,s2)
#print(sample)
t,p = stats.ttest_ind(s1,s2)
p1 = '%f' % (p/2) 
print ("t-statistic:" + str(t))
print("p-value:" + str(p1))

