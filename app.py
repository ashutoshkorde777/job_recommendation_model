import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.markdown("""
    <style>
        
        .stApp {
            background-color: #f4f4f4;
        }

        
        .header {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            background: #4CAF50;
            padding: 15px;
            border-radius: 10px;
        }

       
        .sidebar .sidebar-content {
            background-color: #2E3B4E;
            color: white;
        }

        
        .stTable {
            font-size: 16px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }

       
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    # set correct path
    df = pd.read_csv("merged_normal_vectors.csv")
    return df

df = load_data()

# Extract necessary data
X = df.drop(columns=['O*NET-SOC Code', 'Title'])
Y = df[['O*NET-SOC Code', 'Title']]

# Function to recommend job profiles
def recommend_job(user_skills):
    user_vector = np.array(user_skills).reshape(1, -1)
    similarity = cosine_similarity(user_vector, X)
    top_indices = similarity.argsort()[0][-5:][::-1]  # Get top 5 matches
    return Y.iloc[top_indices][['Title']]

# --- MAIN UI ---
st.markdown('<div class="header">Job Recommendation System</div>', unsafe_allow_html=True)

st.sidebar.header("🔍 Select Your Skill Levels")

# Collect user inputs dynamically
user_vector = []
with st.sidebar.expander("Customize Your Skills ⚙️", expanded=True):
    for col in X.columns:
        value = st.slider(f"{col}", 0.0, 1.0, 0.5, 0.01)
        user_vector.append(value)

# Convert user input to NumPy array
user_vector = np.array(user_vector).reshape(1, -1)

#Recommendations
if True:
    with st.spinner("Finding best matches..."):
        result = recommend_job(user_vector)
        
        # Display results professionally
        st.subheader("🎯 Top Recommended Job Profiles")
        st.table(result)
