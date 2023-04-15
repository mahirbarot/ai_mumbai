import pandas as pd
import streamlit as st
import numpy as np
import pickle
import time


st.set_page_config(
        page_title="A.I end sem project",
        page_icon="money",
        layout='centered',
    )

df=pd.read_csv('mumbai.csv')

st.header("House Price Prediction.")
st.write(df.head())
st.write("")
st.write("")
st.line_chart(data=df,x='Area',y='Price')
X=df.drop('Price',axis=1)
st.write("")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "LinearReg", "89.93")
col2.metric("Accuracy", "DecisionTree", "93.41")
col3.metric("Accuracy", "RandomForest", "97.56")

def predict_price1(Location,Area,bhk,Gymnasium,Lift):

        model=pickle.load(open('rf2.pickle','rb'))

        st.write("")
        st.subheader("Processing inputs...")
        progress=st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)
        st.balloons()


        loc_index = np.where(X.columns==Location)[0][0]

        x = np.zeros(len(X.columns))
        x[0] = Area
        x[1] = bhk
        x[2] = Gymnasium
        x[3] = Lift

        if loc_index >=0:
            x[loc_index] = 1

        #print(x)

        #return model.predict([x])[0]
        ans=model.predict([x])[0]
        ans=round(ans,2)
        st.header(str(ans) +" Lakhs")




df2=pd.read_csv('loc.csv')
loc_val=df2['locs']

loc=st.selectbox("Select location:",loc_val)
area=st.number_input("Enter Area in sqft:",min_value=1,value=455)
bhk=st.slider('Select BHK', 0, 10, 2)
gym= st.radio(
    "Does it have a gymnasium?",
    ('Yes','No'),index=1)

lift= st.radio(
    "Does it have a Lift?",
    ('Yes','No'))


if(gym=='Yes'):
    gym_val=1
else:
    gym_val=0

if(lift=='Yes'):
    lift_val=1
else:
    lift_val=0




if st.button("Predict the price:"):     
        #model=st.selectbox("Choose a model for prediction...",('LinearReg','RandomForest','DecisionTree'),index=1)
        predict_price1(loc,area,bhk,gym_val,lift_val)
        
