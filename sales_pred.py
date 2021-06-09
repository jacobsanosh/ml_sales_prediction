import streamlit as st
import  pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse',False)

prediction_model=open('salespred.pkl','rb')
classifier = pickle.load(prediction_model)

data=pd.read_csv('Advertising.csv')
data=data.drop(['Unnamed: 0'],axis=1)




st.title("let's predict sales")
st.header("adv  heatmap")
sns.heatmap(data.corr(),annot=True)
st.pyplot()


st.header("from the above graph we can undertood that TV ads and sales have highest correlation")

st.header("realtionship between sales and course")
st.write(data.cov())

st.header("correaltion between ads and sales")
st.write(data.corr())
st.write("from the above table we can see that tv ads have high relation")


st.header("describing the data")
st.write(data.describe())


tv_ads=st.number_input("please enter the tv ads")
radio_ads=st.number_input("please enter raadio ads ")
newspaper_ads =st.number_input("please enter the newspaper ")
if tv_ads==0.00 and radio_ads==0.00 and newspaper_ads==0.00:
    st.error("some fields are missing")
else:
    if st.button('predict'):
        prediction=classifier.predict([[tv_ads,radio_ads,newspaper_ads]])
        st.write(prediction)