"""
This app uses a single-step Cosine Similarity approach for career matching. The user responses are compared 
to the career dataset, and the top matches are identified based on similarity scores.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------- Load Data -----------------------
 
# Load career vectors (47 normalized features per career)
career_df = pd.read_csv(r"df_updated_refined_merged.csv")  # Each row is a career with 47 normalized features
X = career_df.drop(columns = ["O*NET-SOC Code", "Title", "Career Number", "Relevant O*NET-SOC Code", "Career Name", "Group Name", "Status Quo with Growth", "Aspirational Career 1", "Aspirational Career 2", "Aspirational Career 3","Aspirational Career 4", "Summary Description"])
Y = career_df[["O*NET-SOC Code", "Title", "Career Number", "Relevant O*NET-SOC Code", "Career Name", "Group Name", "Status Quo with Growth", "Aspirational Career 1", "Aspirational Career 2", "Aspirational Career 3","Aspirational Career 4", "Summary Description"]]
print(X.columns)
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# Load feature categories (mapping each feature to its type)
feature_categories = pd.read_csv("features.csv")  

# Load personality, interests, and self-assessment aptitude questions
aptitude_df = pd.read_csv("aptitude_questions.csv")
interest_df = pd.read_csv("interest_questions.csv")
personality_df = pd.read_csv("personality_questions.csv")

questions_df = pd.concat([aptitude_df, interest_df, personality_df], ignore_index=True) 

# Load aptitude-grades mapping matrix
aptitude_matrix = pd.read_csv("aptitude_grade_structure.csv")  

# Load values ranking scenarios
values_scenarios = pd.read_csv("value_questions.csv")  

# ----------------------- Streamlit UI -----------------------

st.title("🎯 Career Matching System")
st.markdown("""
This app uses a **single-step Cosine Similarity** approach for career matching. The user responses are represented 
as a feature vector, which is compared to the career dataset using Cosine Similarity. The top matches are identified 
based on the highest similarity scores.
""")
st.subheader("Answer the following questions to find your best-fit careers!")

# ----------------------- Personality, Interests, Self-Assessment Aptitude Inputs -----------------------

st.subheader("🧠 Personality, Interests, and Self-Assessment Aptitude")

user_feature_scores = {}

for _, row in questions_df.iterrows():
    question = row["Question"]
    feature = row["Feature"]
    polarity = row["Question Polarity"]
    

    # Slider for user input (0 = strongly disagree, 1 = strongly agree)
    response = st.slider(question, 0.0, 1.0, 0.5)
    
    # Adjust score based on polarity
    user_feature_scores[feature] = user_feature_scores.get(feature, 0) + (response if polarity == "Positive" else (1 - response))

# Normalize to [0,1]
for feature in user_feature_scores:
    user_feature_scores[feature] /= questions_df[questions_df["Feature"] == feature].shape[0]

# ----------------------- Aptitude-Grades Inputs -----------------------

st.subheader("📚 Academic Performance (Grades)")
subject_scores = {}

subjects = ["Mathematics", "Languages", "Science", "Computer science", "Social studies", "Physical Education", "Fine Arts & Design"]

for subject in subjects:
    subject_scores[subject] = st.slider(f"{subject} Score (out of 100)", 0, 100, 50)

# Compute aptitude feature scores based on subject scores
aptitude_feature_scores = {}

for _, row in aptitude_matrix.iterrows():
    feature = row["O*NET Aptitude Level 3"]
    aptitude_feature_scores[feature] = sum(row[subject] * (subject_scores[subject] / 100) for subject in subjects) / 100

# Normalize aptitude scores
max_aptitude_score = max(aptitude_feature_scores.values(), default=1)
for feature in aptitude_feature_scores:
    aptitude_feature_scores[feature] /= max_aptitude_score

user_feature_scores.update(aptitude_feature_scores)

# ----------------------- Values Ranking Inputs -----------------------

st.subheader("💡 Work Values")
value_scores = {}

for _, row in values_scenarios.iterrows():
    scenario = row["Scenarios"]
    options = [row["Answer 1"], row["Answer 2"], row["Answer 3"]]
    values = [row["Value 1"], row["Value 2"], row["Value 3"]]

    st.write(f"**{scenario}**")
    
    # Let user rank the options (each option should be assigned a unique rank)
    ranking = {}
    selected_options = set()

    for rank, score in zip(["1st", "2nd", "3rd"], [4, 2, 0]):  
        choice = st.selectbox(f"Select your {rank} choice:", 
                              [opt for opt in options if opt not in selected_options], 
                              key=f"{scenario}_{rank}")

        ranking[choice] = score
        selected_options.add(choice)

    # Assign scores to values
    for i, val in enumerate(values):
        value_scores[val] = value_scores.get(val, 0) + ranking.get(options[i], 0)

# Normalize values scores
max_value_score = max(value_scores.values(), default=1)
for value in value_scores:
    value_scores[value] /= max_value_score

user_feature_scores.update(value_scores)


# ----------------------- Generate User Vector -----------------------

# Ensure user vector matches the career feature columns
user_vector = np.array([user_feature_scores.get(feature, 0) for feature in X.columns])
print(user_vector)
print(len(user_vector))

# ----------------------- Find Career Matches -----------------------
if st.button("🔍 Find My Career Matches"):
    # Compute similarity scores
    similarities = cosine_similarity(user_vector.reshape(1, -1), X)[0] 

    # Get top 5 indices
    top_indices = np.argsort(similarities)[-5:][::-1]  

    # Retrieve career details
    top_careers = Y.iloc[top_indices][["Title", "Career Name"]].copy()
    top_careers["Similarity"] = similarities[top_indices]

    # Display results
    st.subheader("🎯 Your Top Career Matches")
    for _, row in top_careers.iterrows():
        st.write(f"**{row['Career Name']}** - Similarity Score: {row['Similarity']:.2f}")
