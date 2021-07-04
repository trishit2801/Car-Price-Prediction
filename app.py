import numpy as np
import pandas as pd
import pickle
import streamlit as st

dataset = pd.read_csv('ford.csv')
print(dataset.head())
X = dataset.iloc[:, [0,1,3,4,5,6,7,8]].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
X[:,2] = le.fit_transform(X[:,2])
X[:,4] = le.fit_transform(X[:,4])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train[:,[0,1,3,4,5,6,7]] = sc_X.fit_transform(X_train[:,[0,1,3,4,5,6,7]])
X_test[:,[0,1,3,4,5,6,7]] = sc_X.fit_transform(X_test[:,[0,1,3,4,5,6,7]])
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

pickle_in = open('regressor_model.pkl', 'rb')
regressor = pickle.load(pickle_in)


def predictPrice(model, year, transmission, mileage, fuel, tax, mpg, engine):
    prediction = sc_y.inverse_transform(regressor.predict([[model,year,transmission,mileage,fuel,tax,mpg,engine]]))
    print(prediction)
    return prediction

def main():
      # giving the webpage a title
    st.title("Car Price Prediction")
    
    html_temp = """
    <h2>Enter the below details and get the predicted price of the car</h2>
    """    
    st.markdown(html_temp, unsafe_allow_html = True)
        
    model = st.text_input("Model of the Car", "Type Here")
    year = st.text_input("Year", "Type Here")
    transmission = st.text_input("Transmission type", "Type Here")
    mileage = st.text_input("Mileage", "Type Here")
    fuel = st.text_input("Fuel Type", "Type Here")
    tax = st.text_input("Tax", "Type Here")
    mpg = st.text_input("Miles per gallon", "Type Here")
    engine = st.text_input("Engine Capacity", "Type Here")
    result =""
      
    if st.button("Predict"):
        result = predictPrice(model,year,transmission,mileage,fuel,tax,mpg,engine)
    st.success('Predicted Price of the car is {}'.format(result))

if __name__=='__main__':
    main()