# 假設檢定(總統身高)

import pandas as pd
from scipy import stats
import numpy as np  
df= pd.read_csv('./president_heights.csv')
d=pd.DataFrame(df)
print(d[['height(cm)']])



#last=d.tail(10)
last=d.tail(10)['height(cm)']
np.array(last)
#print(last[['height(cm)']])


#early=df.head(len(df)-10)['height(cm)'] 
early=df.head(len(df)-10)['height(cm)'] 
np.array(early)
#print(early[['height(cm)']])

#s1=last[['height(cm)']]
#s2=early[['height(cm)']]

#sample=np.append(s1,s2)
#print(sample)

#t,p = stats.ttest_ind(s1,s2)
#p1 = '%f' % (p/2) 
#print ("t-statistic:" + str(t))
#print("p-value:" + str(p1))
print('older mean={}, newer mean={}'.format(np.mean(np.array(early)), np.mean(np.array(last))))
tset, pval = stats.ttest_1samp(np.array(last), np.mean(np.array(early)))
pval = pval / 2
print('t-statistic={}'.format(tset))
print('p-values={}'.format(pval))
if pval < 0.05:
    print("有顯著差異")
else:
    print("沒有顯著差異")

