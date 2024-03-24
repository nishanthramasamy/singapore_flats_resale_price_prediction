import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

col1, col2, col3 = st.columns([5,5,5])

with col2:
    st.header(":blue[_Welcome_]")

st.header("", divider='red')

st.write('This site helps you to predict the resale price of the flats in singapore, depending upon your choice of location, no of rooms etc.')
st.write('This model is built on historical data ranging from 1990s to present day')

st.write("_Start predicting_")
df1 = pd.read_csv(rf"https://drive.google.com/file/d/18HkaCMDHRmi3NKddN_4Zu1jM3Q4TIJY-/view?usp=sharing")
df2 = pd.read_csv(rf"https://drive.google.com/file/d/1uzO91O-JWTqfaDd-IqEVwTnOjFv8Vc9b/view?usp=sharing")
df3 = pd.read_csv(rf"https://drive.google.com/file/d/151NQjJCyp3KJNvU0POuegOb9OM8a2fdI/view?usp=sharing")
df4 = pd.read_csv(rf"https://drive.google.com/file/d/1EhdRpjv3ihkPq6SFs76tFZX2ELJNDVfq/view?usp=sharing")
df5 = pd.read_csv(rf"https://drive.google.com/file/d/1YjiciCEJuj4TWuz83i0gweZoFFgEhR2Y/view?usp=sharing")
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

#preprocessing:::::
df[['year', 'months']] = df['month'].str.split('-', expand=True)
df['year'] = pd.to_numeric(df['year'])
df['months'] = pd.to_numeric(df['months'])
df.drop(columns='month', inplace=True)
values = df.columns

features = st.multiselect("Select the values you wish to use to calculate the resale price",values)

features_dict = {
                'town' : "", 
                'street_name': "",
                'block' : "",
                'flat_type': "", 
                'storey_range' : "", 
                'flat_model': "",
                'floor_area_sqm' : "",
                'lease_commence_date': "", 
                'year' : "",
                'months' : ""
}

if features:

    if  'town' in features :
        if 'street_name' in features:
            if'block' in features:
                towns = df['town'].unique().tolist()
                town_selected = st.selectbox("Choose the town ", towns)
                features_dict['town'] = town_selected

                df['street_name'].fillna(method='ffill', inplace=True)
                df['town'].fillna(method='ffill', inplace=True)
                street_names = df['street_name'][df['town'] == town_selected]
                street_name = st.selectbox(f"choose the street form the {town_selected} ", street_names.unique())
                features_dict['street_name'] = street_name
                blocks = df['block'][df['town'] == town_selected][df['street_name'] == street_name]
                block = st.selectbox(f"Choose the block of the street {street_name}", blocks.unique())
                features_dict['block'] = block
            else:
                towns = df['town'].unique().tolist()
                town_selected = st.selectbox("Choose the town ", towns)
                features_dict['town'] = town_selected

                df['street_name'].fillna(method='ffill', inplace=True)
                df['town'].fillna(method='ffill', inplace=True)
                street_names = df['street_name'][df['town'] == town_selected]
                street_name = st.selectbox(f"choose the street form the {town_selected} ", street_names.unique())
                features_dict['street_name'] = street_name
                
        else:
            towns = df['town'].unique().tolist()
            town_selected = st.selectbox("Choose the town ", towns)
            features_dict['town'] = town_selected

    if 'flat_type' in features:

        flat_types = df['flat_type'].unique()
        flat_type = st.selectbox("choose the flat type", flat_types)
        features_dict['flat_type'] = flat_type
    
    if 'storey_range' in features:

        storey_ranges = df['storey_range'].unique()
        storey_range = st.selectbox("choose the storey range", storey_ranges)
        features_dict['storey_range'] = storey_range
    if 'flat_model' in features:

        flat_models = df['flat_model'].unique()
        flat_model = st.selectbox("choose the flat model", flat_models)
        features_dict['flat_model'] = flat_model
    if 'floor_area_sqm' in features:
        floors = df['floor_area_sqm'].unique()
        floor_area = st.selectbox("Choose the floor area", floors)
        features_dict['floor_area_sqm'] = floor_area
    if 'year' in features:
        if 'months' in features:
            years = df['year'].unique()
            year = st.selectbox("Choose the month to check the resale value", years)
            features_dict['year'] = year

            months = df['months'][df['year'] == year].unique()
            month = st.selectbox("Choose the month to check the resale value", months)
            features_dict['months'] = month
        else:
            years = df['year'].unique()
            year = st.selectbox("Choose the month to check the resale value", years)
            features_dict['year'] = year
    if 'lease_commence_date' in features:
        com_dates = df['lease_commence_date'].unique()
        com_date = st.selectbox("choose the lease commencement date", com_dates)
        features_dict['lease_commence_date'] = com_date
    
    
    features = df.copy()
    final_feature = []
    final_column_check = []
    label_encoder = LabelEncoder()

    for col,val in features_dict.items():
        if val:
            final_column_check.append(col)
            features = features[features[col] == val]


    original_columns = df.columns.tolist()

    for i in original_columns:
        if i != 'resale_price':
            if i not in final_column_check:
                features.drop(columns=i, inplace=True)

    predict = st.button("PREDICT")
    if predict:
    
        length = len(features)

        if length < 100:
            st.write("Final model is too minium to predict. Kindly limit the options")
        else:
            
            label_encoder = LabelEncoder()
            for i in features.columns.tolist():
                if i != 'resale_price':
                    features[i] = label_encoder.fit_transform(features[i])

            x = features
            y = features[['resale_price']]


            x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.3, random_state=42)
            decision_model = DecisionTreeRegressor(random_state=42)
            decision_model.fit(x_train, y_train)
            y_pred = decision_model.predict(x_test)
            final_price = "{:.2f}".format(y_pred.mean())
            st.write(f"The resale value of flat of your choice is { final_price} GSD")

