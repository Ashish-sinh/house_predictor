import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import streamlit  as st 
import pickle 

dataset = pickle.load(open('/house_dataset.pkl','rb')) 
scaler = pickle.load(open('/scaler.pkl','rb')) 
model = pickle.load(open('/regression_model.pkl','rb')) 


st.title('🏠 House Price Predictor') 
st.write('Please Enter Below Details To predict the Price of Your House')
st.image('download.jpeg') 
        
        # Details from Datasets : 
        #  - MedInc        median income in block group
        # - HouseAge      median house age in block group
        # - AveRooms      average number of rooms per household
        # - AveBedrms     average number of bedrooms per household
        # - Population    block group population
        # - AveOccup      average number of household members
        # - Latitude      block group latitude
        # - Longitude     block group longitude
        

Median_income = st.number_input('Enter the Median Income' , min_value= dataset.MedInc.values.min() ,  max_value= dataset.MedInc.values.max() ,step = 2.0 )
HouseAge = st.number_input('Enter the House Age ' , min_value= dataset.HouseAge.values.min() ,  max_value= dataset.HouseAge.values.max() ,step = 10.0 )

averooms , avebedrms  = st.columns(2) 
AveRooms = averooms.number_input('Enter the Average Number of Rooms' , min_value= dataset.AveRooms.values.min() ,  max_value= dataset.AveRooms.values.max() ,step = 2.0 )
AveBedrms= avebedrms.number_input('Enter the Average Number of BedRooms' , min_value= dataset.AveBedrms.values.min() ,  max_value= dataset.AveBedrms.values.max() ,step = 2.0 )

Population = st.number_input('Enter the Population ' , min_value= dataset.Population.values.min() ,  max_value= dataset.Population.values.max() ,step = 50.0 )
AveOccup = st.number_input('Enter the Average Number of HouseHold Members' , min_value= dataset.AveOccup.values.min() ,  max_value= dataset.AveOccup.values.max() ,step = .50 )

latitude ,longitude = st.columns(2) 
Latitude = latitude.slider('Enter the Latitude Income' , min_value= dataset.Latitude.values.min() ,  max_value= dataset.Latitude.values.max())
Longitude = longitude.slider('Enter the Longitude Income' , min_value= dataset.Longitude.values.min() ,  max_value= dataset.Longitude.values.max()) 



my_input = np.array([Median_income , HouseAge , AveRooms , AveBedrms , Population , AveOccup , Latitude , Longitude]).reshape(1,-1) 

scaled_input = scaler.transform(my_input) 
output = model.predict(scaled_input) 

if st.button('Predict House Price') : 
    st.title(f'Predicted House Price in $ : {output[0]* 10000}')
