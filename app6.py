"""
This app uses an Autoencoder-based dimensionality reduction technique combined with Approximate Nearest Neighbors (ANN) 
using FAISS for career matching. The user responses are transformed into a lower-dimensional space, and the closest 
career matches are identified using L2 distance in the encoded space.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import faiss  # Add this import for ANN
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os
import optuna  # Add this import for hyperparameter tuning


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
This app uses an **Autoencoder** for dimensionality reduction and **Approximate Nearest Neighbors (ANN)** with FAISS 
for career matching. The Autoencoder is a neural network that compresses input data into a lower-dimensional latent 
space and reconstructs it back. FAISS is used to find the closest matches in this reduced space using L2 distance.
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
    response = st.radio(
        row["Question"],
        options=list(response_options.keys()),
        key=f"question_{row.name}"
    )
    score = response_options[response] if row["Question Polarity"] == "Positive" else (1 - response_options[response])
    user_feature_scores[row["Feature"]] = user_feature_scores.get(row["Feature"], 0) + score

for feature in user_feature_scores:
    user_feature_scores[feature] /= questions_df[questions_df["Feature"] == feature].shape[0]

# ----------------------- Aptitude Inputs -----------------------
st.subheader("📚 Academic Performance (Grades)")
subject_scores = {subj: st.slider(f"{subj} (out of 100)", 0, 100, 50) for subj in aptitude_matrix.columns[3:]}  


aptitude_feature_scores = {}
for _, row in aptitude_matrix.iterrows():
    feature = row["O*NET Aptitude Level 3"]
    aptitude_feature_scores[feature] = sum(
        float(int(row[subj])) * (subject_scores[subj] / 100)
        for subj in subject_scores if subj in row
    ) / 100


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


# Apply Weights
user_vector_weighted = user_vector

# ----------------------- Autoencoder Definition -----------------------
input_dim = X.shape[1]
encoding_dim = 32  # Dimension of the encoded representation

# Define the path to save/load the model
model_dir = os.path.dirname(__file__)  # Directory of the current script
model_path = os.path.join(model_dir, "autoencoder_model.h5")  # Save model in the same directory as the script

# ----------------------- Hyperparameter Tuning -----------------------
def objective(trial):
    # Define hyperparameters to tune
    encoding_dim = trial.suggest_int("encoding_dim", 16, 128, step=16)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 10, 100, step=10)

    # Define the autoencoder model
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the autoencoder
    history = autoencoder.fit(
        X.values, X.values,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,  # Suppress output for tuning
        validation_split=0.2
    )

    # Return the validation loss as the objective value
    return min(history.history['val_loss'])

# Run Optuna study
if not os.path.exists(model_path):
    st.write("🔍 Tuning hyperparameters...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Get the best hyperparameters
    best_params = study.best_params
    st.write(f"Best Hyperparameters: {best_params}")

    # Train the final autoencoder with the best hyperparameters
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(best_params["encoding_dim"], activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mean_squared_error')
    autoencoder.fit(
        X.values, X.values,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        shuffle=True,
        verbose=1
    )

    # Save the trained model
    autoencoder.save(model_path)
    encoder = Model(autoencoder.input, autoencoder.layers[-2].output)  # Extract encoder
else:
    # Load the saved model if it exists
    autoencoder = load_model(model_path)
    encoder = Model(autoencoder.input, autoencoder.layers[-2].output)  # Extract encoder

# Transform job vectors using the trained encoder
X_transformed = encoder.predict(X.values)

# ----------------------- Transform User Vector -----------------------
user_vector_transformed = encoder.predict(user_vector.reshape(1, -1))

# ----------------------- ANN Index Creation -----------------------
# Create a FAISS index for transformed job vectors
dimension = X_transformed.shape[1]  # Encoded dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance metric

# Ensure the transformed vectors are C-contiguous and of type float32
X_transformed_contiguous = np.ascontiguousarray(X_transformed, dtype=np.float32)
faiss.normalize_L2(X_transformed_contiguous)  # Normalize the transformed vectors
index.add(X_transformed_contiguous)  # Add transformed job vectors to the index

# ----------------------- Career Matching -----------------------
if st.button("🔍 Find My Career Matches"):
    # Normalize the transformed user vector
    user_vector_transformed_contiguous = np.ascontiguousarray(user_vector_transformed, dtype=np.float32)
    faiss.normalize_L2(user_vector_transformed_contiguous)

    # Perform ANN search on transformed vectors
    _, top_indices = index.search(user_vector_transformed_contiguous, 5)  # Top 5 matches
    top_indices = top_indices[0]  # Extract indices

    # Display top job matches
    st.subheader("🎯 Your Top Career Matches")
    for idx in top_indices:
        st.write(f"**{Y.iloc[idx]['Career Name']}**")
