import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from joblib import dump, load
@st.cache
st.text('Adesh Kumar\n05.09.2021')
def load_data():
    sc = load('state_city.pkl')
    model = load('model.pkl')
    onehencoder = load('onehencoder.pkl')
    std_norm = load('std_scaler.pkl')
    return sc, model, onehencoder, std_norm

sc, model, onehencoder, std_norm = load_data()

st.title('Credit Risk Detection')
st.header('Make Prediction')
st.subheader('Customer\'s feature')
state = st.selectbox(
    'State',
    (['Andhra Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Mizoram', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttar Pradesh[5]', 'Uttarakhand', 'West Bengal']))
city = st.selectbox(
    'City',
    (sorted(sc[state])))
profession = st.selectbox(
    'Profession',
    ('Air traffic controller', 'Analyst', 'Architect', 'Army officer', 'Artist', 'Aviator', 'Biomedical Engineer', 'Chartered Accountant', 'Chef', 'Chemical engineer', 'Civil engineer', 'Civil servant', 'Comedian', 'Computer hardware engineer', 'Computer operator', 'Consultant', 'Dentist', 'Design Engineer', 'Designer', 'Drafter', 'Economist', 'Engineer', 'Fashion Designer', 'Financial Analyst', 'Firefighter', 'Flight attendant', 'Geologist', 'Graphic Designer', 'Hotel Manager', 'Industrial Engineer', 'Lawyer', 'Librarian', 'Magistrate', 'Mechanical engineer', 'Microbiologist', 'Official', 'Petroleum Engineer', 'Physician', 'Police officer', 'Politician', 'Psychologist', 'Scientist', 'Secretary', 'Software Developer', 'Statistician', 'Surgeon', 'Surveyor', 'Technical writer', 'Technician', 'Technology specialist', 'Web designer'))
married = st.radio(
    "Marriage Status",
    (['married', 'single']))
house_ownership = st.radio(
    "House Ownership",
    ('norent_noown', 'owned', 'rented'))
car_ownership = st.radio(
    "Car Ownership",
    ('no', 'yes'))
age = st.number_input("Age", max_value= 79, min_value = 21)
experience = st.number_input("Experience", max_value= 21, min_value = 0)
current_job_years = st.number_input("Current Job Years", max_value= 15, min_value = 0)
current_house_years = st.number_input("Current House Years", max_value= 14, min_value = 10)
income = st.number_input("Income", min_value = 10310, max_value = 9999938)
if st.button('Submmit'):
    d = {'state': state,
         'city': city,
         'profession': profession,
         'married': married,
         'house_ownership': house_ownership,
         'car_ownership' : car_ownership,
         'age': age,
         'experience': experience,
         'current_job_years': current_job_years,
         'current_house_years': current_house_years,
         'income': income
         }
    data_pred = pd.DataFrame(d, index = [0])
    data_pred['profession'] = data_pred['profession'].str.replace(' ','_')
    data_pred['profession'] = data_pred['profession'].str.lower()
    data_pred['state'] = data_pred['state'].str.replace(' ','_')
    data_pred['state'] = data_pred['state'].str.lower()
    data_pred['city'] = data_pred['city'].str.replace(' ','_')
    data_pred['city'] = data_pred['city'].str.lower()
    encode_features = ['state','city','profession','married','car_ownership','house_ownership']
    data_encode = onehencoder.transform(data_pred[encode_features])
    std_features = ['income','age','experience','current_job_years','current_house_years']
    data_norm = std_norm.transform(data_pred[std_features])
    data = hstack((data_encode, data_norm)).tocsr()
    pred = model.predict(data)
    st.subheader('Prediction Result')
    st.write('Feature used')
    st.write(d)
    if pred == 1:
        st.write('**Risk Flage :**', 'Defaulter')
    else:
        st.write('**Risk Flage :**', 'Non Defaulter')
else:
    st.write('For Prediction click on submmit button')
