import streamlit as st  
import plotly.graph_objects as go  
import time  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report  
from datetime import datetime
import git

# -------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")  

st.title('Streamlit Data Science Demo')  

tabs = ["Feature Importance", "Status", "Query Builders"]  
tab = st.sidebar.radio("Select a tab", tabs)  

# -------------------------------------------------------------------------------------------------------
#spotify data
#load data
spotify_df = pd.read_csv("/Users/camilaaichele/Desktop/app/csv_data_files/playlist_2010to2022.csv")  
numerical_spotify_df = spotify_df.loc[:,['year', 'track_popularity', 'artist_popularity',  
                        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',  
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',  
                        'duration_ms', 'time_signature']]  
#drop NaN values
numerical_spotify_df.dropna(inplace=True) 
#features vs target
selected_features1 = ['year','track_popularity','danceability', 'energy', 'key',  
                'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',  
                'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']  
target1 = ['artist_popularity']
# Calculate the correlation matrix
correlation_matrix1 = numerical_spotify_df.corr()

# -------------------------------------------------------------------------------------------------------
#breast cancer data
# Load the data  
breast_cancer_df = pd.read_csv("/Users/camilaaichele/Desktop/app/csv_data_files/Breast_cancer_data.csv")  
#drop NaN values
breast_cancer_df.dropna(inplace=True)  
#features vs target
selected_features2 = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
                                'mean_smoothness']  
target2 = ['diagnosis']  
# Calculate the correlation matrix
correlation_matrix2 = breast_cancer_df.corr()

# -------------------------------------------------------------------------------------------------------
#cardiovascular data
# Load the data  
cardio_df = pd.read_csv("/Users/camilaaichele/Desktop/app/csv_data_files/cardio_data_processed.csv")
#drop NaN values
cardio_df.dropna(inplace=True)  
#features vs target
selected_features3 = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'age_years',
                                'bmi']  
target3 = ['bp_category_encoded'] 
# Calculate the correlation matrix
# Calculate the correlation matrix
correlation_matrix3 = cardio_df.corr(numeric_only=True)  # Add the numeric_only parameter

# -------------------------------------------------------------------------------------------------------
#status data
# Initialize an empty list to store status updates
if "status_updates" not in st.session_state:
    st.session_state.status_updates = []
    
# -------------------------------------------------------------------------------------------------------
if tab == "Feature Importance":  
    st.header("Feature Importance")  
    if st.session_state.get('tab1', True):  
        def_selectbox = st.selectbox(  
            "Choose a table (dataset) to view their models (feature importance model)",  
            ("Spotify", "Breast Cancer", "Cardiovascular Disease")  
        )  
# -------------------------------------------------------------------------------------------------------
        if def_selectbox == "Spotify":  

            X = numerical_spotify_df[selected_features1]  
            y = numerical_spotify_df[target1]  

            X = pd.get_dummies(X)  

            # Train-Test split  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

            # Build the Random Forest Classifier  
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  

            # Train set  
            rf_classifier.fit(X_train, y_train)  

            # Predict on the test set  
            y_pred = rf_classifier.predict(X_test)  

            # Get feature importances  
            feature_importances = rf_classifier.feature_importances_  

            # Create a bar chart for feature importances (sorted in descending order)  
            sorted_indices = feature_importances.argsort()[::-1]  # Sort indices in descending order  
            sorted_features = X_train.columns[sorted_indices]  
            sorted_importances = feature_importances[sorted_indices]  

            st.header("Sample Data")
            st.write(spotify_df.head())
            
            st.header("Export the models as figures")
            # Create the bar chart  
            fig1, ax1 = plt.subplots(figsize=(6,3))  
            ax1.barh(sorted_features, sorted_importances, color='skyblue')  
            ax1.set_title('Feature Importances artist_popularity')  
            ax1.set_xlabel('Importance')  
            ax1.set_ylabel('Features')  

            # Display the chart in Streamlit  
            st.pyplot(fig1)  

            # Display the correlation matrix as a heatmap
            fig_corr1, ax_corr1 = plt.subplots(figsize=(8, 4))
            sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr1)
            plt.title("Correlation Matrix")
            st.pyplot(fig_corr1)     
            
            st.header("Interactive")
            # Create a Line Plot of Track Popularity over the Years
            st.subheader("Track Popularity over the Years")
            popularity_over_years = spotify_df.groupby('year')['track_popularity'].mean()
            st.line_chart(popularity_over_years)

            # Create a Bar Plot of Artist Genres
            st.subheader("Artist Genres")
            genre_counts = spotify_df['artist_genres'].str.split(',').explode().str.strip().value_counts()
            st.bar_chart(genre_counts.head(10))
            
            st.subheader("Feature Importances")
            st.bar_chart(sorted_importances)
            
            print("hi")
            
# -------------------------------------------------------------------------------------------------------
        elif def_selectbox == "Breast Cancer":   

            X = breast_cancer_df[selected_features2]  
            y = breast_cancer_df[target2]  

            X = pd.get_dummies(X)  

            # Train-Test split  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

            # Build the Random Forest Classifier  
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  

            # Train set  
            rf_classifier.fit(X_train, y_train)  

            # Predict on the test set  
            y_pred = rf_classifier.predict(X_test)  

            # Get feature importances  
            feature_importances = rf_classifier.feature_importances_  

            # Create a bar chart for feature importances (sorted in descending order)  
            sorted_indices = feature_importances.argsort()[::-1]  # Sort indices in descending order  
            sorted_features = X_train.columns[sorted_indices]  
            sorted_importances = feature_importances[sorted_indices]  

            st.header("Sample Data")
            st.write(breast_cancer_df.head())            
            
            st.header("Export the models as figures")
            # Create a Pairplot
            st.subheader("Pairplot")
            pairplot1 = sns.pairplot(breast_cancer_df, hue="diagnosis")
            st.pyplot(pairplot1.fig)
            
            # Create the bar chart  
            fig2, ax2 = plt.subplots(figsize=(8,4))  
            ax2.barh(sorted_features, sorted_importances, color='skyblue')  
            ax2.set_title('Feature Importances diagnosis')  
            ax2.set_xlabel('Importance')  
            ax2.set_ylabel('Features')  

            # Display the chart in Streamlit  
            st.pyplot(fig2)  

            # Display the correlation matrix as a heatmap
            fig_corr2, ax_corr2 = plt.subplots(figsize=(8, 4))
            sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr2)
            plt.title("Correlation Matrix")
            st.pyplot(fig_corr2)
            
            st.header("Interactive")
            # Create a Bar Plot of Diagnosis Counts
            st.subheader("Diagnosis Counts")
            diagnosis_counts = breast_cancer_df['diagnosis'].value_counts()
            st.bar_chart(diagnosis_counts)
            
            st.subheader("Feature Importances")
            st.bar_chart(sorted_importances)
# -------------------------------------------------------------------------------------------------------
        elif def_selectbox == "Cardiovascular Disease":  

            X = cardio_df[selected_features3]  
            y = cardio_df[target3]  

            # Train-Test split  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

            # Build the Random Forest Classifier  
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  

            # Train set  
            rf_classifier.fit(X_train, y_train)  

            # Predict on the test set  
            y_pred = rf_classifier.predict(X_test)  

            # Get feature importances  
            feature_importances = rf_classifier.feature_importances_  

            # Create a bar chart for feature importances (sorted in descending order)  
            sorted_indices = feature_importances.argsort()[::-1]  # Sort indices in descending order  
            sorted_features = X_train.columns[sorted_indices]  
            sorted_importances = feature_importances[sorted_indices]          
            
            st.header("Sample Data")
            st.write(cardio_df.head())
            
            st.header("Export the models as figures")
            # Create a Pairplot
            st.subheader("Pairplot")
            pairplot2 = sns.pairplot(cardio_df, hue="bp_category_encoded")
            # Display the chart in Streamlit
            st.pyplot(pairplot2.fig)
            
            # Blood Pressure Category Distribution
            fig31, ax31 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=cardio_df, x='bp_category', ax=ax31)
            plt.title('Blood Pressure Category Distribution')
            plt.xlabel('Blood Pressure Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)

            # Display the chart in Streamlit
            st.pyplot(fig31)

            # Create the bar chart  
            fig3, ax3 = plt.subplots(figsize=(8, 4))  
            ax3.barh(sorted_features, sorted_importances, color='skyblue')  
            ax3.set_title('Feature Importances bp_category')  
            ax3.set_xlabel('Importance')  
            ax3.set_ylabel('Features')  

            # Display the chart in Streamlit  
            st.pyplot(fig3)    

            # Display the correlation matrix as a heatmap
            fig_corr3, ax_corr3 = plt.subplots(figsize=(8, 4))
            sns.heatmap(correlation_matrix3, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr3)
            plt.title("Correlation Matrix")
            st.pyplot(fig_corr3)
            
            st.header("Interactive")
            # Create a Bar Plot of Cholesterol vs. Cardio
            st.subheader("Cholesterol vs. Cardio")
            cholesterol_counts = cardio_df['cholesterol'].value_counts()
            st.bar_chart(cholesterol_counts)

            # Create a Bar Plot of Gender vs. Cardio
            st.subheader("Gender vs. Cardio")
            gender_counts = cardio_df['gender'].value_counts()
            st.bar_chart(gender_counts)
            
            st.subheader("Feature Importances")
            st.bar_chart(sorted_importances)
            
            print("hi")
            
# -------------------------------------------------------------------------------------------------------            
        with st.spinner("Loading..."):  
            time.sleep(5)  
        st.success("Done!")  
# -------------------------------------------------------------------------------------------------------
    else:  
        st.session_state['tab1'] = False  
# -------------------------------------------------------------------------------------------------------
elif tab == "Status":
    st.header("Status")
    
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add the current status update to the list
    status_update = f"{current_datetime} - Status update"
    
    # Fetch the latest Git commit message (code changes)
    repo_path = "https://github.com/camila1014/streamlistapptest"
    
    try:
        repo = git.Repo(repo_path)
        latest_commit = repo.head.commit
        code_changes = latest_commit.message.strip()
        print("Code Changes:", code_changes)
    except Exception as e:
        code_changes = "Unable to fetch code changes"
        print("Error:", e)


    # Display the latest code changes
    st.subheader("Latest Code Changes:")
    st.write(code_changes)

    
    # Display the text box to input what needs to be fixed
    st.subheader("Enter What Needs to Be Fixed:")
    fixes_needed = st.text_area("Type here...", key="fixes_needed")
    
    # Create a button to save the content of the text areas into session state
    if st.button("Save Status"):
        
        # Include the fixes_needed content in the status update if not empty
        if fixes_needed:
            status_update += f"\n  - Fixes Needed: {fixes_needed}"
        
        st.session_state.status_updates.append(status_update)
    
    # Display all status updates
    st.subheader("Progress Report:")
    for idx, update in enumerate(st.session_state.status_updates, 1):
        st.write(f"{idx}. {update}")
    
# -------------------------------------------------------------------------------------------------------
elif tab == "Query Builders":    
    st.header("Query Builders")  
    fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[8, 4, 2])])  
    st.plotly_chart(fig)  
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)  