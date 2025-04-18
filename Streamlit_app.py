import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


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



# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Skippy\ml-matching\csv_files\df_updated_refined_merged.csv")
    questions_df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Skippy\ml-matching\csv_files\questions.csv")  # Load questions
    return df, questions_df

df, questions_df = load_data()

# Extract feature vectors
X = df.drop(columns = ["O*NET-SOC Code", "Title", "Career Number", "Relevant O*NET-SOC Code", "Career Name", "Group Name", "Status Quo with Growth", "Aspirational Career 1", "Aspirational Career 2", "Aspirational Career 3","Aspirational Career 4", "Summary Description"])
Y = df[["O*NET-SOC Code", "Title", "Career Number", "Relevant O*NET-SOC Code", "Career Name", "Group Name", "Status Quo with Growth", "Aspirational Career 1", "Aspirational Career 2", "Aspirational Career 3","Aspirational Career 4", "Summary Description"]]

# Normalize feature vectors
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# Define Likert scale mapping
likert_mapping = {
    "Strongly Disagree": 0.0,
    "Disagree": 0.25,
    "Neutral": 0.5,
    "Agree": 0.75,
    "Strongly Agree": 1.0
}

# ---- UI ----
st.markdown('<div class="header">Job Recommendation System</div>', unsafe_allow_html=True)
st.sidebar.header("🔍 Select Your Skill Levels Based on Questions")

# --- Store user responses ---
user_responses = {}

# Group questions by feature
grouped_questions = questions_df.groupby("Level 2 Dimension")

# Identify features that don't have questions
existing_features = set(questions_df["Level 2 Dimension"])
missing_features = set(X.columns) - existing_features

# Iterate over each feature and its questions
with st.sidebar.expander("Customize Your Skills ⚙️", expanded=True):
    for feature, questions in grouped_questions:
        st.subheader(f"🔹 {feature}")
        positive_scores = []
        negative_scores = []

        for _, row in questions.iterrows():
            response = st.radio(row['Question'], options=list(likert_mapping.keys()), index=2, key=row['Question'])
            score = likert_mapping[response]

            if row["Dimension Question #"] == 1.0:  # Positive Dimension
                positive_scores.append(score)
            elif row["Dimension Question #"] == 2.0:  # Negative Dimension
                negative_scores.append(1 - score)  # Invert the score

        # Compute the final feature score (average of both dimensions)
        all_scores = positive_scores + negative_scores
        user_responses[feature] = np.mean(all_scores) if all_scores else 0.5

    # Add missing features with default questions
    for feature in missing_features:
        st.subheader(f"🔹 {feature}")
        default_question = f"When I start working, I would like to have {feature}"
        response = st.radio(default_question, options=list(likert_mapping.keys()), index=2, key=default_question)
        user_responses[feature] = likert_mapping[response]


# Convert user responses to match X's feature order
user_vector = [user_responses.get(feature, 0.5) for feature in X.columns]
user_vector = np.array(user_vector).reshape(1, -1)

# ---- Recommendation Function ----
def recommend_job(user_skills):
    similarity = cosine_similarity(user_skills, X)
    top_indices = similarity.argsort()[0][-5:][::-1]  # Get top 5 matches
    return Y.iloc[top_indices][['Title',"Career Name"]]

# ---- Show Recommendations ----
if True:
    with st.spinner("Finding best matches..."):
        print(user_vector)
        print("vector length: ",len(user_vector[0]))
        result = recommend_job(user_vector)
        print(result)
       
        
        st.subheader("🎯 Top Recommended Job Profiles")
        st.table(result)