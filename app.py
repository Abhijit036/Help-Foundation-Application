import streamlit as st
import numpy as np
import pandas as pd
import joblib

# First lets load the instances that were created

with open('scaler.joblib','rb') as file:
    scale=joblib.load(file)

with open('pca.joblib','rb') as file:
    pca=joblib.load(file)

with open('final_model.joblib','rb') as file:
    model=joblib.load(file)

def prediction(input_list):
    scaled_input=scale.transform([input_list])
    pca_input=pca.transform(scaled_input)
    output=model.prediction(pca_input)[0]

    if output == 0:
        return 'Developed'
    elif output == 1:
        return 'UnderDeveloped'
    else:
        return 'Developing'

def main():
    st.title('Help NGO Foundation')
    st.subheader('This application will given status of a country based on socio-economic and health factors.')

    gdp=st.text_input('Enter the GDP per Population of a country')
    inc=st.text_input('Enter the per capita income of the country')
    imp=st.text_input('Enter the Imports in terms of % of GDP')
    exp=st.text_input('Enter the Exports in terms of % of GDP')
    inf=st.text_input('Enter the inflation rate of the country(%)')
    hel=st.text_input('Enter the expenditure on health in terms of % of GDP')
    ch_m=st.text_input('Enter the no. of deaths per 1000 birth for <5 yrs')
    fer=st.text_input('Enter the avg Children born to a women')
    lf=st.text_input('Enter the average life expectancy in a country')

    in_data=[ch_m,exp,hel,imp,inc,inf,lf,fer,gdp]

    if st.button('Predict'):
        response = predict(in_data)
        st.success(response)

if __name__=='__main__':
    main()
