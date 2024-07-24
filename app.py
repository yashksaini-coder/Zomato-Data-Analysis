import streamlit as st
# Importing the basic libraries
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the tasks for each level
tasks = {
    'Level 1': ['Task 1', 'Task 2', 'Task 3'],
    'Level 2': ['Task 1', 'Task 2', 'Task 3'],
    'Level 3': ['Task 1', 'Task 2', 'Task 3']
}

# Create the Streamlit web app
def main():
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
        
        st.write('---')    

if __name__ == '__main__':
    main()