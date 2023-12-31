import streamlit as st
import pandas as pd
import sklearn
import pickle as pk
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def checkNotNull(Credit_amt, population, distance, hour, history_30, interaction_30):
    if Credit_amt == 0 or population == 0 or distance == 0 or hour == 0 or history_30 == 0 or interaction_30 == 0:
        st.error("Field Should Not be Zero")
        return True
    else:
        return False
    
def Prediction(x):
    
    with open('decision_model.pkl', 'rb') as file:
        loaded_model = pk.load(file) 
    
    prediction = loaded_model.predict(x.reshape(1,-1))
    #st.header(prediction)
    return prediction



def transformf(Credit_amt, population, distance, hour, history_30, interaction_30):
    arr = np.array([Credit_amt,population, distance,hour,history_30, interaction_30])
    #st.header(arr)
    return arr



def scaling(arr):
    # Scaling
    #st.header(arr)
    #from sklearn.preprocessing import StandardScaler, MinMaxScaler
    #sc = StandardScaler()
# Independent Features
    #x = sc.fit_transform(arr)
    #st.header(x)
    import numpy as np

    sample = np.array([845, 65, 87, 56])
    mean = sample.mean()
    std = sample.std()
    scaled_sample = (sample - mean) / std

    st.header(scaled_sample)
    return scaled_sample


def run():
    
    st.title("🏦 CREDIT CARD FRAUD DETECTION 📈")
    st.balloons()
    # Set background color and padding for the title
    st.markdown(
        """
        <style>
            .title {
                color: white;
                background-color: #007ACC;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="title"><h1>CREDIT CARD FRAUD DETECTION 📈</h1></div>', unsafe_allow_html=True)

    # Add a horizontal line separator
    st.markdown("---")

    st.write("Fill In The Required Information to Predict")

    # Add input fields for the required features with custom styling
    Credit_amt = st.number_input("💰 Credit Card Amount", step=1, value=0)
    population = st.number_input("🌍 Population", step=1, value=0)
    distance = st.number_input("🏢 Distance", step=1, value=0)
    hour = st.number_input("🌎 Hour", step=1, value=0)
    history_30 = st.number_input("🏢  30 Days Payment History", step=1, value=0)
    interaction_30 = st.number_input("🏙️ 30 Days Payment Interaction", step=1, value=0)
    credit_card_number = st.number_input("💳 Credit Card Number", step=1, value=0)
    with st.container():
        st.caption("Above Details Are Transformed Attributes for Machine Learning Model for the Prediction")
        st.caption("Not Actual Attributes Which Are Received From Bank Data")


    # Apply styling to the submit button
    submit_button_style = "background-color: #007ACC; color: white; padding: 10px; border-radius: 5px; text-align: center;"
    if st.button("🚀 Submit", key="submit", help="Click to predict loan approval"):
        # Prepare the features for prediction
        res = checkNotNull(Credit_amt, population, distance, hour, history_30, interaction_30)
        arr = transformf(Credit_amt, population, distance, hour, history_30, interaction_30)
        #x = scaling(arr)

        prediction = Prediction(arr)

        if prediction[0] == 1 and res == False:
            st.error("❌ Credit Card Payment is Fraud for the Card Number ")
            st.error(credit_card_number)
        elif res == False:
            st.success("✅ Credit Card Payment is Not Fraud for the Card Number ")
            st.success(credit_card_number)
    st.divider()
    st.text("Team 14 | Data Commanders")


run()


#np.array([841.623325,	595,	0.754073,	16,	4063.2 17256,4.839003]).reshape(1,-1)

