import pandas as pd
import numpy as np
import joblib
import streamlit as st

bike_model_pkl = r'C:\Users\vansh\Desktop\PC\ML\CareerEra\ML\models\bike_share_randomforest.pkl'
loaded_model = joblib.load(bike_model_pkl)

st.header("Bike Rental Prediction")

# season=st.text_input("Enter the season ")
season = st.selectbox("Enter the Season", ("Spring", "Summer", "Autumn", "Winter"))
season_dict={"Spring":1, "Summer": 2, "Fall": 3, "Winter":4}
season=season_dict[season]


month=st.selectbox("Enter the month", ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"))
months = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}
month = months[month]



holiday=st.selectbox("Enter the holiday", ("Public Holiday", "Not a Holiday"))
holiday_dict={"Public Holiday":1, "Not a Holiday":0}
holiday=holiday_dict[holiday]

weekday=st.selectbox("Enter the weekday ", ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
weekday_dict = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}
weekday=weekday_dict[weekday]

workingday=st.selectbox("Enter the workingday ", ("Working Day", "Not a Working Day"))
workingday_dict={"Working Day":1, "Not a Working Day":0}
workingday=workingday_dict[workingday]


weathersit=st.selectbox("Enter the weathersit ", ("Clear", "Mist/Cloudy", "Light Rainfall/Snowfall", "Hailstorm"))
weathersit_dict={"Clear":1, "Mist/Cloudy":2, "Light Rainfall/Snowfall":3, "Hailstorm":4}
weathersit=weathersit_dict[weathersit]

temp=st.number_input("Enter the temperature ")/75
atemp=st.number_input("Enter the apparent temperature ")/75

hum=st.number_input("Enter the humidity ")/75
windspeed=st.number_input("Enter the windspeed ")/75



X_new = np.array([[season,month,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed]])

button = st.button("Submit")

if button:

    result = loaded_model.predict(X_new)
    result=np.round(result[0])

    st.info("Total Number of rentals : " + str(result))