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


    #st.write("_Start predicting_")
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


with tab2:

    #Function to encode categorical variables::
    def get_encoded(df, data):
        encoder = LabelEncoder()
        encoder.fit(df)
        encoded_column = encoder.transform(df)
        encoded_data = encoder.transform([data])

        return encoded_column, encoded_data[0]

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

    #selecting the features
    towns = df['town'].unique().tolist()
    town_selected = st.selectbox("Choose the town ", towns)
    features_dict['town'] = town_selected
    df['encoded_town'], features_dict['town'] = get_encoded(df['town'],town_selected)



    flat_types = df['flat_type_modified'].unique()
    flat_type = st.selectbox("choose the flat type", flat_types)
    df['encoded_flat_type'], features_dict['flat_type'] = get_encoded(df['flat_type_modified'], flat_type)
        


    flat_models = df['flat_model'].unique()
    flat_model = st.selectbox("choose the flat model", flat_models)
    df['encoded_flat_model'], features_dict['flat_model'] = get_encoded(df['flat_model'], flat_model)
        
        
    floors = df['floor_area_sqm'].unique()
    floor_area = st.text_input("Choose the floor area (eg: 45.00, 67.65) in sq meter")
    if floor_area:
        floor_area = int(floor_area)
    else:
        floor_area = df['floor_area_sqm'].mean()
    features_dict['floor_area_sqm'] = int(floor_area)


    years = df['year'].unique()
    year = st.selectbox("Choose the year of contruction", years)
    features_dict['year'] = year


    leases = df['remaining_lease'].unique()
    lease = st.text_input("choose the remaining years of lease (Eg: 96, 45)")
    if lease:
        lease = int(lease)
    else:
        lease = 0
    features_dict['remaining_lease'] = int(lease)

    x = df[['encoded_town', 'encoded_flat_type', 'encoded_flat_model', 'floor_area_sqm', 'year', 'remaining_lease']]
    y = df['resale_price']


    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
    dt_regressor = RandomForestRegressor()
    decision_model = RandomForestRegressor(random_state=42)
    decision_model.fit(x_train, y_train)

    #PICKLING:::
    with open("model.pk1", "wb") as file:
        pickle.dump(decision_model, file)

    with open("model.pk1", "rb") as file:
        unloaded = pickle.load(file)

    y_pred_random = unloaded.predict(x_test)
    mae_random = mean_absolute_error(y_pred_random, y_test)
    r2score_random = r2_score(y_pred_random, y_test)

    predict = st.button("PREDICT")
    if predict:            

        
            #st.write("randomforest mae", mae)

            #Predicting unknown choice from user
            x_final = []
            #st.write("features_dict.values=", features_dict.values())
            for i in list(features_dict.values()):
                if i :
                    x_final.append(i)
            #st.write("x_final= ", x_final) 
            final_price = decision_model.predict([x_final])
            final_price = round(final_price[0])
            st.write(f"The resale value of flat of your choice is :blue[{final_price}] GSD")


# with tab3:
#     st.write("This section gives you information behind the model. Check below for more details")


#     st.header("Parameter Selection:")
#     st.write("The dataset had many features like street name, blocks, town etc. But we use certain features which help in creating model.")

#     st.write("Click below to visualise relationship between features")

#     st1, st2, st3, st4 = st.columns([5,5,5,5])
#     with st1:
#         visual = st.button("Click here to see the charts")
   
#      #EDA::: Visualising the features with respect to target variable to understand the relationship
#     if visual:
#         col1, col2= st.columns([5,5])
#         with col1:
#             fig, ax = plt.subplots()
#             sns.scatterplot(data=df, x='year', y='resale_price', ax=ax)
#             plt.xticks(rotation=90)
#             st.pyplot(fig)
#         with col2:
#             fig, ax = plt.subplots()
#             plt.figure(figsize=(10, 6))
#             sns.scatterplot(data=df, x='flat_model', y='resale_price', ax=ax)
#             plt.xticks(rotation=90)
#             st.pyplot(fig)
#         col3, col4 = st.columns([5,5])
#         with col3:
#             fig, ax = plt.subplots()
#             plt.figure(figsize=(10, 6))
#             sns.scatterplot(data=df, x='town', y='resale_price', ax=ax)
#             plt.xticks(rotation=90)
#             st.pyplot(fig)
#         with col4:
#             fig, ax = plt.subplots()
#             sns.scatterplot(data=df, x='flat_type_modified', y='resale_price', ax=ax)
#             plt.xticks(rotation=90) 
#             st.pyplot(fig)

#     st.write("Feature importance:")
#     importance_scores = decision_model.feature_importances_
#     scores = []
#     col = features_dict.keys
#     for x in importance_scores:
#         scores.append(x)
#     data = {"features":["Town", "Flat type", "Flat model", "Floor Area", "Year", "Remaining Lease"],
#             "importance score": scores}
#     sc = pd.DataFrame(data)
#     st.dataframe(sc)

#     st.header("Model used:")
#     #DecisionTree:::
#     x = df[['encoded_town', 'encoded_flat_type', 'encoded_flat_model', 'floor_area_sqm', 'year', 'remaining_lease']]
#     y = df['resale_price']
#     x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
#     decision_model = DecisionTreeRegressor(random_state=42)
#     decision_model.fit(x_train, y_train)
#     y_pred = decision_model.predict(x_test)
#     mae = mean_absolute_error(y_pred, y_test)
#     r2 = r2_score(y_test, y_pred)
    

#     st.write("The model is trained on :orange[RandomForestRegressor]. Initially :orange[LinearRegression], :orange[DecisionTreeRegressor] where tried but their performance was low compared to Randomforest")
#     st.write(f"DecisionTree : R^2 Score: {r2:.2f}")
#     st.write("Decision Tree - MAE:", mae)

#     #RandomForest:
#     st.write(f"MAE - RandomForest: {mae_random}")
#     st.write(f"R^2 Score - RandomForest: {r2score_random:.2f}")

