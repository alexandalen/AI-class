## Titanic predict survived by Streamlit

# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas as pd

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# load model
random_forest = load('model.joblib')
scaler = load('std.joblib')

def convert_sex(sex1):
    return 1 if sex1 == '男性' else 0

def convert_age(age1):
    bins = [0, 15, 35, 60, 100]
    return pd.cut([age1], bins, labels=[3,1,2,0])[0]

def convert_embark_town(embark1):
    dict1 = {'Southampton':1, 'Cherbourg':2, 'Queenstown':3}
    return dict1[embark1] 



# 畫面設計
st.markdown("# 鐵達尼號乘客生存評估系統")
pclass_series = pd.Series([1, 2, 3])
sex_series = pd.Series(['male', 'female'])
embark_town_series = pd.Series(['Cherbourg', 'Queenstown', 'Southampton'])

#下拉是選單
sex = st.sidebar.selectbox('性別:', sex_series)
age = st.sidebar.slider('年齡', 0, 100, 20)
sibsp = st.sidebar.slider('同行人數', 0, 10, 0)
embark_town = st.sidebar.selectbox('上船港口:', embark_town_series)
parch = st.sidebar.slider('父母子女同行人數', 0, 10, 0)
pclass = st.sidebar.selectbox('船艙等級:',pclass_series)
fare = st.sidebar.slider('船票價錢:', 0, 100, 20)

if st.sidebar.button('預測'):
    '性別:', sex
    '年齡:', age
    '兄弟姊妹同行人數:', sibsp
    '父母子女同行人數:', parch
    '上船港口:', embark_town
    '艙等:', pclass
    '票價:', fare

    # predict
    X = []
    # pclass	sex	age	sibsp	parch	fare	embark_town
    X.append([pclass, convert_sex(sex), convert_age(age), sibsp, parch, fare, convert_embark_town(embark_town)])
    X=np.array(X)
    print(X)
    X = scaler.transform(X)
    # print(X)
    if random_forest.predict(X) == 1:
        st.markdown('==> **生存**')
    else:
        st.markdown('==> **死亡**')