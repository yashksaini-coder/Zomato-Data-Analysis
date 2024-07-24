import streamlit as st
# Importing the basic libraries
import numpy as np
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

    st.title(f'{selected_level}')
    for task in tasks[selected_level]:
        
        if task == 'Task 1' and selected_level == 'Level 1':
            ### Task 1: Data Exploration and Preprocessing
            st.markdown('### Task 1: Data Exploration and Preprocessing')
            st.write('---')
            
            st.write('Load the dataset and identify the number of rows and columns.')
            df = pd.read_csv("./data/data.csv")
            st.write(df.head())
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            st.write('---')
            
            st.write('Check for missing values in each column and handle them.')
            st.write(df.isnull().sum())
            st.write('---')
            
            st.write('Perform data type conversion if necessary.')
            df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(int)
            st.write("The Data processed is:-\n",df['Votes'])
            st.write('---')
            
            st.write('Analyse the distribution of the target variable (“Aggregate rating”) and identify any class imbalances.')    
            df_aggregate = df[(df['Aggregate rating']>0)&(df['Votes']>0)]
            plt.figure(figsize=(10, 6))
            sns.histplot(df_aggregate['Aggregate rating'], bins=20, kde=True, palette='viridis')
            plt.title('Distribution of Aggregate Rating')
            plt.xlabel('Aggregate Rating')
            plt.ylabel('Frequency')
            st.write(plt.show())

        if task == 'Task 2' and selected_level == 'Level 1':
            st.markdown('### Task 2: Descriptive Analysis')
            
            st.write('1. Calculate basic statistical measures (mean, median, standard deviation, etc.) for numerical columns.')
            st.write(df.describe())
            
            st.write('2. Explore the distribution of categorical variables like "Country Code," "City," and "Cuisines". ')
            st.write('---')
            st.write('3. Identify the top cuisines and cities with the highest number of restaurants.')
            st.write('---')
    st.write('---')    

if __name__ == '__main__':
    main()