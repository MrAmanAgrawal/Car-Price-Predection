import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

def index():
    st.title('Car Price Predictor')
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    company = st.selectbox('Select Company', companies)
    car_model = st.selectbox('Select Car Model', car_models)
    year = st.selectbox('Select Year', year)
    fuel_type = st.selectbox('Select Fuel Type', fuel_type)
    kms_driven = st.number_input('Enter KMs driven', min_value=1, value=1)

    if st.button('Predict Price'):
        prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type ]], columns=['name', 'company','year', 'kms_driven', 'fuel_type']))
        st.success(f'Predicted Price is {np.round(prediction[0], 2)}')

if __name__=='__main__':
    index()
