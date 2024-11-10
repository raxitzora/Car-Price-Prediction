import streamlit as st
import pandas as pd
import pickle

# Load the data and model
data = pd.read_csv("Cleaned data.csv")
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

# Extract the unique car names and companies from the dataset
car_names = sorted(data['name'].unique())
car_companies = sorted(data['company'].unique())

# Define the prediction function
def predict_price(name, company, year, kms_driven, fuel_type):
    # Create an input DataFrame for the model
    input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    # Make predictions
    prediction = model.predict(input_data)
    return prediction[0]

# Build the Streamlit app interface
st.title("Car Price Prediction")

# Input options for the user
name = st.selectbox("Select Car Model", car_names)
company = st.selectbox("Select Car Company", car_companies)
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2024, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
fuel_type = st.selectbox("Select Fuel Type", ["Petrol", "Diesel", "CNG"])

# Predict button
if st.button("Predict Price"):
    try:
        price = predict_price(name, company, year, kms_driven, fuel_type)
        st.success(f"Estimated Price: ₹{price:,.2f}")
    except Exception as e:
        st.error("Error: Unable to predict price. Please check your inputs and try again.")
        st.error(str(e))
