import streamlit as st
# Importing the basic libraries
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go


# Define the tasks for each level
tasks = {
    'Level 1': ['Task 1', 'Task 2', 'Task 3'],
    'Level 2': ['Task 1', 'Task 2', 'Task 3'],
    'Level 3': ['Task 1', 'Task 2', 'Task 3']
}

# Create the Streamlit web app
def main():
    st.title('Zomato Data Analysis')
    
    st.sidebar.title('Navigation')
    selected_level = st.sidebar.selectbox('Select Level', list(tasks.keys()))
    selected_task = st.sidebar.selectbox('Select Task', tasks[selected_level])

    st.title(f'{selected_level} - {selected_task}')
    
    if selected_task == 'Task 1' and selected_level == 'Level 1':
        ### Task 1: Data Exploration and Preprocessing
        st.markdown('### Task 1: Data Exploration and Preprocessing')
        st.write('---')
        
        st.markdown('- Load the dataset and identify the number of rows and columns.')
        df = pd.read_csv("./data/data.csv")
        st.write(df.head())
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write('---')
        
        st.markdown('- Check for missing values in each column and handle them.')
        st.write(df.isnull().sum())
        st.write('---')
        
        st.markdown('- Perform data type conversion if necessary.')
        df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(int)
        st.write("The Data processed is:-\n",df['Votes'])
        st.write('---')
        
        st.markdown('- Analyse the distribution of the target variable (“Aggregate rating”) and identify any class imbalances.')    
        df_aggregate = df[(df['Aggregate rating']>0)&(df['Votes']>0)]
        plt.figure(figsize=(10, 6))
        sns.histplot(df_aggregate['Aggregate rating'], bins=20, kde=True, palette='viridis')
        plt.title('Distribution of Aggregate Rating')
        plt.xlabel('Aggregate Rating')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    if selected_task == 'Task 2' and selected_level == 'Level 1':
        st.markdown('### Task 2: Descriptive Analysis')
        df = pd.read_csv("./data/data.csv")
        st.markdown('- Calculate basic statistical measures (mean, median, standard deviation, etc.) for numerical columns.')
        st.write(df.describe())
        
        # st.write('2. Explore the distribution of categorical variables like "Country Code," "City," and "Cuisines". ')            
        st.markdown("- Explore the distribution of `Country Code` categorical variables")
        plt.figure(figsize=(14, 7))
        sns.countplot(y='Country Code', data=df, order=df['Country Code'].value_counts().index)
        plt.title('Distribution of Country Code')
        st.pyplot(plt)
        
        st.markdown("- Explore the distribution of `City` categorical variables")
        plt.figure(figsize=(14, 7))
        sns.countplot(y='City', data=df, order=df['City'].value_counts().index[:20]) # Top 20 cities for better visualization
        plt.title('Distribution of City')
        st.pyplot(plt)
        
        st.markdown("- Explore the distribution of `Cuisines` categorical variables")            
        plt.figure(figsize=(14, 7))
        sns.countplot(y='Cuisines', data=df, order=df['Cuisines'].value_counts().index[:20]) # Top 20 cuisines for better visualization
        plt.title('Distribution of Cuisines')
        st.pyplot(plt)

        st.write('---')
        st.markdown('- Identify the top cuisines and cities with the highest number of restaurants.')
        # Identify the top cuisines and cities with the highest number of restaurants
        top_cuisines = df['Cuisines'].value_counts().head(10)
        top_cities = df['City'].value_counts().head(10)
        st.write('Top 10 Cuisines:', top_cuisines)
        st.write('Top 10 Cities:', top_cities)
        
    if selected_task == 'Task 3' and selected_level == 'Level 1':
        df = pd.read_csv("./data/data.csv") 
        st.markdown("### Task 3: Geospatial Analysis")
        
        st.markdown("- Visualize the locations of restaurants on a map using latitude and longitude information.")
        plt.figure(figsize=(14, 7))
        fig = px.scatter_geo(df, lat='Latitude', lon='Longitude', hover_name='Restaurant Name', color='Aggregate rating',title='Restaurant Locations and Ratings')
        st.plotly_chart(fig)
        
        st.write('---')        
        
        st.markdown("- Analyse the distribution of restaurants across different cities or countries.")
        plt.figure(figsize=(14, 7))
        fig = px.scatter_geo(df, lat='Latitude', lon='Longitude', hover_name='City', color='City',title='Restaurant Distribution by City')
        st.plotly_chart(fig)
        
        st.write('---')
        
        st.markdown("- Determine if there is any correlation between the restaurant's location and its rating.")
        plt.figure(figsize=(14, 7))
        sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Aggregate rating', palette='coolwarm')
        plt.title('Correlation between Location and Rating')
        st.pyplot(plt)

        st.write('---')
        
    if selected_task == 'Task 1' and selected_level == 'Level 2':
        st.markdown("### Task 1: Table Booking and Online Delivery")
        
        df = pd.read_csv("./data/data.csv")
        
        st.markdown("- Determine the percentage of restaurants that offer table booking")
        table_booking_percentage = df['Has Table booking'].value_counts(normalize=True) * 100
        st.write('Percentage of restaurants that offer table booking:\n', table_booking_percentage)

        st.markdown("- Determine the percentage of restaurants that offer online booking")
        online_delivery_percentage = df['Has Online delivery'].value_counts(normalize=True) * 100
        st.write('Percentage of restaurants that offer online delivery:\n', online_delivery_percentage)
        st.write('---')
        
        st.markdown("- Compare the average ratings of restaurants with table booking and those without.")
        avg_rating_table_booking = df.groupby('Has Table booking')['Aggregate rating'].mean()
        st.write('Average rating of restaurants with/without table booking:\n', avg_rating_table_booking)
        st.write('---')
        
        st.markdown("- Analyse the availability of online delivery among restaurants with different price ranges.")
        online_delivery_price_range = df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack() * 100
        st.write('Online delivery availability by price range:\n', online_delivery_price_range)
        st.write('---')
        
        st.markdown("- Determine the percentage of restaurants that offer table booking.")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=table_booking_percentage.index, y=table_booking_percentage.values)
        plt.title('Percentage of Restaurants Offering Table Booking')
        plt.xlabel('Has Table Booking')
        plt.ylabel('Percentage')
        st.pyplot(plt)
        st.write('---')
        
        st.markdown("- Determine the percentage of restaurants that offer online delivery.")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=online_delivery_percentage.index, y=online_delivery_percentage.values)
        plt.title('Percentage of Restaurants Offering Online Delivery')
        plt.xlabel('Has Online Delivery')
        plt.ylabel('Percentage')
        st.pyplot(plt)
        st.write('---')
        
        st.markdown("- Compare the average ratings of restaurants with table booking and those without.")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=avg_rating_table_booking.index, y=avg_rating_table_booking.values)
        plt.title('Average Rating of Restaurants with/without Table Booking')
        plt.xlabel('Has Table Booking')
        plt.ylabel('Average Rating')
        st.pyplot(plt)
        st.write('---')
        
        st.markdown("- Analyse the availability of online delivery among restaurants with different price ranges.")
        plt.figure(figsize=(10, 6))
        online_delivery_price_range.plot(kind='bar', stacked=True)
        plt.title('Online Delivery Availability by Price Range')
        plt.xlabel('Price Range')
        plt.ylabel('Percentage')
        plt.legend(title='Has Online Delivery', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)
        st.write('---')
        
    if selected_task == 'Task 2' and selected_level == 'Level 2':
        st.markdown("### Task 2: Price Range Analysis")
        
        df = pd.read_csv("./data/data.csv")
        
        st.markdown("- Determine the most common price range among all the restaurants")
        most_common_price_range = df['Price range'].mode()[0]
        st.write('Most common price range:', most_common_price_range)
        st.write('---')
        
        st.markdown("- Calculate the average rating for each price range")
        avg_rating_price_range = df.groupby('Price range')['Aggregate rating'].mean()
        st.write('Average rating for each price range:\n', avg_rating_price_range)
        st.write('---')
        
        st.markdown("- Identify the colour that represents the highest average rating among different price ranges.")
        color_avg_rating = df.groupby('Price range')['Rating color'].agg(lambda x: x.mode()[0])
        st.write('Color representing the highest average rating for each price range:\n', color_avg_rating)
        st.write('---')
    
        st.markdown("- Visualize the distribution of price range among all the restaurants.")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Price range', data=df, order=df['Price range'].value_counts().index)
        plt.title('Distribution of Price Range')
        plt.xlabel('Price Range')
        plt.ylabel('Count')
        st.pyplot(plt)
        st.write('---')
        
        st.markdown("- Identify the most common price range among all the restaurants.")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=avg_rating_price_range.index, y=avg_rating_price_range.values)
        plt.title('Average Rating by Price Range')
        plt.xlabel('Price Range')
        plt.ylabel('Average Rating')
        st.pyplot(plt)
        st.write('---')
        
        st.markdown("- Identify the colour that represents the highest average rating among different price ranges.")
        color_rating_df = pd.DataFrame({'Price range': avg_rating_price_range.index, 'Rating color': color_avg_rating.values})
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Price range', y='Price range', data=color_rating_df, hue='Rating color', dodge=False)
        plt.title('Color Representing the Highest Average Rating by Price Range')
        plt.xlabel('Price Range')
        plt.ylabel('Average Rating')
        st.pyplot(plt)
        st.write('---')
        
    if selected_task == 'Task 3' and selected_level == 'Level 2':        
        st.markdown("### Task 3: Feature Engineering")
        df2 = pd.read_csv("./data/data.csv")
        st.write('---')
        
        st.markdown("- Extract additional features from the existing columns, such as the length of the restaurant name or address.")
        df['Restaurant Name Length'] = df2['Restaurant Name'].apply(len)
        df['Address Length'] = df2['Address'].apply(len)
        st.write('Extracted Features:\n', df2[['Restaurant Name Length', 'Address Length']])
        
        st.write('---')
        
        st.markdown("- Create new features like `Has Table Booking` or `Has Online Delivery` by encoding categorical variables.")
        df2['Has Table Booking'] = df2['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
        df2['Has Online Delivery'] = df2['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        st.write('New Features:\n', df2.head())

        st.markdown("- Analyse the distribution of the newly created features.")        
        plt.figure(figsize=(10, 6))
        sns.histplot(df2['Restaurant Name Length'], bins=20, kde=True)
        plt.title('Distribution of Restaurant Name Length')
        plt.xlabel('Restaurant Name Length')
        plt.ylabel('Frequency')
        st.pyplot(plt)
        st.write('---')
   
        plt.figure(figsize=(10, 6))
        sns.histplot(df2['Address Length'], bins=20, kde=True)
        plt.title('Distribution of Address Length')
        plt.xlabel('Address Length')
        plt.ylabel('Frequency')
        st.pyplot(plt)
        st.write('---')

        st.markdown("- Analyse the count of Table Booking.")        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Has Table Booking', data=df2)
        plt.title('Count of Restaurants with/without Table Booking')
        plt.xlabel('Has Table Booking')
        plt.ylabel('Count')
        st.pyplot(plt)
        st.write('---')
        
        
        st.markdown("- Analyse the count of Online Delivery.")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Has Online Delivery', data=df2)
        plt.title('Count of Restaurants with/without Online Delivery')
        plt.xlabel('Has Online Delivery')
        plt.ylabel('Count')
        st.pyplot(plt)
        st.write('---')

    if selected_task == 'Task 1' and selected_level == 'Level 3':
        df = pd.read_csv("./data/data.csv")
        
        st.markdown("### Task 1: Predictive Modelling")

        st.markdown(" **Preprocess the data**: Handle categorical variables and normalize the data.")
        
        le = LabelEncoder()
        df['Cuisines'] = le.fit_transform(df['Cuisines'])
        df['City'] = le.fit_transform(df['City'])
        df['Country Code'] = le.fit_transform(df['Country Code'])
        df['Rating color'] = le.fit_transform(df['Rating color'])
        df['Has Table booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['Has Online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)
        st.write('Data after preprocessing:\n', df.head())
        
        
        # Select features and target variable
        features = ['Country Code', 'City', 'Cuisines', 'Price range', 'Has Table booking', 'Has Online delivery']
        X = df[features]
        y = df['Aggregate rating']
        
        # Splitting the data into Train & Test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write('---')
        
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.markdown("- **Build and evaluate models**: Train and evaluate different regression models.")
        # Train and evaluate different regression models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        
        # Displaying the models
        st.write("The models used are:-\n",models)
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R2': r2}
            
        st.write('---')
        # Display results
        results_df = pd.DataFrame(results).T
        st.write("After predication the scores are:-\n",results_df)
        st.write('---')
        

        
if __name__ == '__main__':
    main()