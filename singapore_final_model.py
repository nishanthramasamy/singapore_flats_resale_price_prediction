import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

tab1, tab2 = st.tabs(["Home", "Calculate"])

with tab1:

    col1, col2, col3 = st.columns([5,5,5])

    with col2:
        st.header(":blue[_Welcome_]")

    st.header("", divider='red')

    st.write('This site helps you to calculate the resale price of the flats in singapore, depending upon your choice of location, no of rooms etc.')
    st.write('This model is built on historical data ranging from 1990s to present day')
    st.write("Flats value go up and down depending upon several factors like Area of presence, year when was it built and so on.")
    st.write("This model gives an accurate prediction of flats price factoring in several important parameters.")
    st.write("Check out the calculate tab!!!!!")


df1 = pd.read_csv(rf"data_1990_1999.csv")
df2 = pd.read_csv(rf"data_2000_2012.csv")
df3 = pd.read_csv(rf"data_2012_2014.csv")
df4 = pd.read_csv(rf"data_2015_2016.csv")
df5 = pd.read_csv(rf"data_2017_present.csv")
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

#preprocessing && Feature Engineering
df['resale_price'].fillna(df['resale_price'].mean(), inplace=True)
df['town'].fillna(method='ffill', inplace=True)
df['flat_type_modified'] = [x.replace('-', ' ') for x in df['flat_type']]

df['year'] = [x.split('-')[0] for x in df['month']]
df['year'] = pd.to_numeric(df['year'])
df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'])

df['remaining_lease'].fillna(99 - (df['year'] - df['lease_commence_date']), inplace=True)
df['remaining_lease'] = df['remaining_lease'].astype(str).str.split(' ').str[0]
df['remaining_lease'] = pd.to_numeric(df['remaining_lease'])


def get_encoded(data, df):
    encoder = LabelEncoder()
    encoder.fit(df)
    # encoded_column = encoder.transform(df)
    encoded_data = encoder.transform([data])

    return  encoded_data[0]

values = df.columns

features = []

features_dict = {
                'town' : "",
                'flat_type': "",  
                'flat_model': "",
                'floor_area_sqm' : "",
                'year' : "",
                'remaining_lease': ""
}

with tab2:
    try:
#selecting the features
        towns = df['town'].unique().tolist()
        town_selected = st.selectbox("Choose the town ", towns)
        if town_selected:
            features_dict['town'] = get_encoded(town_selected, df['town'])
        else:
            features_dict['town'] = get_encoded("WOODLANDS", df['town'])



        flat_types = df['flat_type_modified'].unique()
        flat_type = st.selectbox("choose the flat type", flat_types)
        if flat_type:
            features_dict['flat_type'] = get_encoded(flat_type, df['flat_type_modified'])
        else:
            features_dict['flat_type'] = get_encoded("3 ROOM", df['flat_type_modified'])
        
            


        flat_models = df['flat_model'].unique()
        flat_model = st.selectbox("choose the flat model", flat_models)
        if flat_model:
            features_dict['flat_model'] = get_encoded(flat_model, df['flat_model'])
        else:
            features_dict['flat_model'] = get_encoded("STANDARD", df['flat_model'])
            
            
        floors = df['floor_area_sqm'].unique()
        floor_area = st.text_input("Choose the floor area (eg: 45.00, 67.65) in sq meter")
        if floor_area:
            floor_area = int(floor_area)
        else:
            floor_area = df['floor_area_sqm'].mean()
        features_dict['floor_area_sqm'] = int(floor_area)


        years = df['year'].unique()
        year = st.selectbox("Choose the year of contruction", years)
        if year:
            features_dict['year'] = year
        else:
            features_dict['year'] = 2000


        leases = df['remaining_lease'].unique()
        lease = st.text_input("choose the remaining years of lease (Eg: 96, 45)")
        if lease:
            lease = int(lease)
        else:
            lease = 0
        features_dict['remaining_lease'] = int(lease)


        loaded_model = joblib.load('random_model.pkl.gz')

        predict = st.button("PREDICT")
        if predict:            

            #Predicting unknown choice from user
            x_final = []
            for i in list(features_dict.values()):
                if i :
                    x_final.append(i)

            final_price = loaded_model.predict([x_final])
            final_price = round(final_price[0])
            st.write(f"The resale value of flat of your choice is :blue[{final_price}] GSD")
    except:
        st.write("Kindly Enter all the fields to calculate the price")
