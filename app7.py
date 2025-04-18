"""
This app uses a two-step Cosine Similarity approach for career matching. Primary features are used for the 
first round of similarity computation, and secondary features refine the results to provide the top career matches.

Methodology:
1. **Data Preparation**:
   - Load career data and feature categories.
   - Normalize features using MinMaxScaler.
   - Group features into categories: "Aptitude-Grades", "Interest", "Aptitude-Self-Assessment", "Personality", and "Values".

2. **User Input**:
   - Collect user responses for personality, interest, and aptitude questions.
   - Gather academic performance (grades) and work value rankings.

3. **Feature Scaling**:
   - Compute feature scores for the user based on inputs.
   - Scale features independently within their respective categories.

4. **Cosine Similarity**:
   - Perform a two-step similarity computation:
     a. Primary features: Identify top matches using primary categories.
     b. Secondary features: Refine matches using secondary categories.

5. **Results**:
   - Display top career matches with similarity scores.
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

# Custom CSS for aligning radio buttons in one horizontal line
st.markdown("""
    <style>
        div[data-baseweb="radio"] > div {
            display: flex;
            flex-direction: row;
            gap: 15px;
        }
        div[data-baseweb="radio"] label {
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Load Data -----------------------
career_df = pd.read_csv("df_updated_refined_merged.csv")
feature_columns = [col for col in career_df.columns if col not in ["O*NET-SOC Code", "Title", "Career Number", "Relevant O*NET-SOC Code", "Career Name", "Group Name", "Status Quo with Growth", "Aspirational Career 1", "Aspirational Career 2", "Aspirational Career 3","Aspirational Career 4", "Summary Description"]]
X = career_df[feature_columns]
Y = career_df[["O*NET-SOC Code", "Title", "Career Number", "Relevant O*NET-SOC Code", "Career Name", "Group Name", "Status Quo with Growth", "Aspirational Career 1", "Aspirational Career 2", "Aspirational Career 3","Aspirational Career 4", "Summary Description"]]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

feature_categories = pd.read_csv("features.csv")  
aptitude_df = pd.read_csv("aptitude_questions.csv")
interest_df = pd.read_csv("interest_questions.csv")
personality_df = pd.read_csv("personality_questions.csv")
values_scenarios = pd.read_csv("value_questions.csv")  

# Add Category column to each DataFrame
aptitude_df["Category"] = "Aptitude"
interest_df["Category"] = "Interest"
personality_df["Category"] = "Personality"

# Concatenate all question DataFrames
questions_df = pd.concat([aptitude_df, interest_df, personality_df], ignore_index=True)

aptitude_matrix = pd.read_csv("aptitude_grade_structure.csv")  

# ----------------------- Feature Weights -----------------------
WEIGHTS = {
    "Aptitude-Grades": 0.4,
    "Aptitude-Self-Assessment": 0.4,
    "Personality": 0.2,
    "Interest": 0.3,
    "Values": 0.1
}


st.title("🎯 Career Matching Model")
st.markdown("""
This app uses a **two-step Cosine Similarity** approach for career matching. In the first step, primary features are 
used to compute similarity scores and identify potential matches. In the second step, secondary features refine the 
matches.
""")
st.subheader("Answer the questions to find your best-fit careers!")

# ----------------------- Personality, Interests, Aptitude Inputs -----------------------
user_feature_scores = {}

response_options = {
    "Strongly Disagree": 0.0,
    "Disagree": 0.25,
    "Neutral": 0.5,
    "Agree": 0.75,
    "Strongly Agree": 1.0
}

for _, row in questions_df.iterrows():
    question_type = row["Category"]  # Use the Category column to get the question type
    question_text = f"[{question_type}] {row['Question']}"  # Add the type to the question text
    response = st.radio(
        question_text,
        options=list(response_options.keys()),
        key=f"question_{row.name}"
    )
    score = response_options[response] if row["Question Polarity"] == "Positive" else (1 - response_options[response])
    user_feature_scores[row["Feature"]] = user_feature_scores.get(row["Feature"], 0) + score

for feature in user_feature_scores:
    user_feature_scores[feature] /= questions_df[questions_df["Feature"] == feature].shape[0]

# ----------------------- Aptitude Inputs -----------------------
st.subheader("📚 Academic Performance (Grades)")

# Collect subject scores
subject_scores = {subj: st.slider(f"{subj} (out of 100)", 0, 100, 50) for subj in aptitude_matrix.columns[3:]}

# Identify top 2 subjects dynamically
sorted_subjects = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
top_2_subjects = [subj for subj, _ in sorted_subjects[:2]]  # Top 2 subjects
remaining_subjects = [subj for subj in subject_scores if subj not in top_2_subjects]

# Apply scaling weights
scaled_subject_scores = {}
for subj, score in subject_scores.items():
    if subj in top_2_subjects:
        scaled_subject_scores[subj] = score * 1.5  # High weight for top 2 subjects
    else:
        scaled_subject_scores[subj] = score * 0.8  # Low weight for remaining subjects

# Compute aptitude feature scores
aptitude_feature_scores = {}
for _, row in aptitude_matrix.iterrows():
    feature = row["O*NET Aptitude Level 3"]
    aptitude_feature_scores[feature] = sum(
        float(int(row[subj])) * (scaled_subject_scores[subj] / 100)
        for subj in scaled_subject_scores if subj in row
    ) / 100

# Normalize aptitude feature scores
max_aptitude_score = max(aptitude_feature_scores.values(), default=1)
for feature in aptitude_feature_scores:
    aptitude_feature_scores[feature] /= max_aptitude_score

user_feature_scores.update(aptitude_feature_scores)

# ----------------------- Values Ranking Inputs -----------------------
st.subheader("💡 Work Values")
value_scores = {}
for _, row in values_scenarios.iterrows():
    options = [row["Answer 1"], row["Answer 2"], row["Answer 3"]]
    ranking = {}
    selected_options = set()
    for rank, score in zip(["1st", "2nd", "3rd"], [4, 2, 0]):
        choice = st.selectbox(f"Select your {rank} choice:", [opt for opt in options if opt not in selected_options], key=f"{row['Scenarios']}_{rank}")
        ranking[choice] = score
        selected_options.add(choice)
    for i, val in enumerate([row["Value 1"], row["Value 2"], row["Value 3"]]):
        value_scores[val] = value_scores.get(val, 0) + ranking.get(options[i], 0)

max_value_score = max(value_scores.values(), default=1)
for value in value_scores:
    value_scores[value] /= max_value_score

user_feature_scores.update(value_scores)

print(user_feature_scores)

# ----------------------- Generate User Vector -----------------------
user_vector = np.array([user_feature_scores.get(feature, 0) for feature in X.columns])

# Scale user vector features independently by category
scaled_user_vector = np.zeros_like(user_vector)
for category in feature_categories["Category"].unique():
    category_features = feature_categories[feature_categories["Category"] == category]["Feature"].tolist()
    if category_features:
        indices = [X.columns.get_loc(feature) for feature in category_features if feature in X.columns]
        category_values = user_vector[indices]
        if len(category_values) > 0:
            # Apply MinMax scaling to the category-specific features
            category_min = category_values.min()
            category_max = category_values.max()
            if category_max - category_min > 0:  # Avoid division by zero
                category_scaled = (category_values - category_min) / (category_max - category_min)
            else:
                category_scaled = category_values  # If all values are the same, no scaling is needed
            scaled_user_vector[indices] = category_scaled

user_vector = scaled_user_vector  # Replace the original user vector with the scaled one
print(user_vector)
print(X.columns)

# ----------------------- Feature Categories -----------------------
# Define feature categories for sequential cosine similarity
primary_categories = ["Aptitude-Grades", "Interest","Aptitude-Self-Assessment"]
secondary_categories = ["Personality", "Values"]

# Filter features based on categories
primary_features = feature_categories[feature_categories["Category"].isin(primary_categories)]["Feature"].tolist()
secondary_features = feature_categories[feature_categories["Category"].isin(secondary_categories)]["Feature"].tolist()

# ----------------------- Sequential Cosine Similarity -----------------------
if st.button("🔍 Find My Career Matches"):
    # Step 1: Cosine similarity based on primary features
    X_primary = X[primary_features].fillna(0)  # Replace NaN values in X_primary with 0
    user_vector_primary = pd.DataFrame([user_feature_scores], columns=X.columns).fillna(0)[primary_features]  # Replace NaN in user vector with 0

    primary_similarities = cosine_similarity(user_vector_primary, X_primary)[0]
    top_primary_indices = np.argsort(primary_similarities)[-20:][::-1]  # Top 10 matches from primary features

    # Display top jobs after the first iteration
    st.subheader("🔍 Top Jobs After Primary Features Matching")
    top_primary_careers = Y.iloc[top_primary_indices].copy()
    top_primary_careers["Similarity"] = primary_similarities[top_primary_indices]
    for _, row in top_primary_careers.iterrows():
        st.write(f"**{row['Career Name']}** - Similarity Score: {row['Similarity']:.2f}")

    # Filter the dataset to include only the top matches from primary similarity
    X_secondary = X[secondary_features].iloc[top_primary_indices].fillna(0)  # Replace NaN values in X_secondary with 0
    Y_filtered = Y.iloc[top_primary_indices]

    # Step 2: Cosine similarity based on secondary features
    user_vector_secondary = pd.DataFrame([user_feature_scores], columns=X.columns).fillna(0)[secondary_features]  # Replace NaN in user vector with 0
    secondary_similarities = cosine_similarity(user_vector_secondary, X_secondary)[0]

    # Combine primary and secondary similarity scores using weighted sum
    combined_similarities = (0.75 * primary_similarities[top_primary_indices]) + (0.25 * secondary_similarities)

    # Get top 5 matches based on combined similarity scores
    top_final_indices = np.argsort(combined_similarities)[-5:][::-1]
    top_careers = Y_filtered.iloc[top_final_indices].copy()
    top_careers["Similarity"] = combined_similarities[top_final_indices]

    # Display results
    st.subheader("🎯 Your Top Career Matches")
    for _, row in top_careers.iterrows():
        st.write(f"**{row['Career Name']}** - Similarity Score: {row['Similarity']:.2f}")
