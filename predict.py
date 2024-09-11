import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Load the data and model
data = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Car_prediction\\Cleaned data.csv")
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

# Get the unique car names and companies from the data and sort them
car_names = sorted(data['name'].unique())
car_companies = sorted(data['company'].unique())

# Define a function to make predictions
def predict_price(name, company, year, kms_driven, fuel_type):
    input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    return model.predict(input_data)[0]

# Build the Streamlit interface
st.title("Car Price Prediction")

# Let the user select car name and company from dropdowns
name = st.selectbox("Car Model Name", options=car_names)
company = st.selectbox("Car Company", options=car_companies)

# Other inputs
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2024, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])

# Predict button and result display
if st.button("Predict Price"):
    price = predict_price(name, company, year, kms_driven, fuel_type)
    st.write(f"Estimated Price: ₹{price:,.2f}")
