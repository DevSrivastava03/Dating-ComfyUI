import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import random

SERVER = "hype2.pig-herring.ts.net:8188"
WORKFLOW_NAME = "workflow_example.json"
client_id = str(uuid.uuid4())

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('speeddating 2.csv')
    threshold = 0.5  # Drop columns with more than 50% missing values
    data = data.loc[:, data.isnull().mean() < threshold]
    
    # Impute missing numerical values with median
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    return data

# Define matchmaking function to suggest a single match
def suggest_single_match(user_profile, model, X_columns, scaler, X_data, y_data):
    # Create a full DataFrame with zeros for all columns
    user_profile_full = pd.DataFrame(0, index=[0], columns=X_columns)
    
    # Update with user's actual profile values
    for col in user_profile.columns:
        if col in user_profile_full.columns:
            user_profile_full[col] = user_profile[col]
    
    # Scale the user profile and all data
    user_data_scaled = scaler.transform(user_profile_full)
    X_data_scaled = scaler.transform(X_data)
    
    # Predict match probabilities for all data points
    match_probs = model.predict_proba(X_data_scaled)[:, 1]
    
    # Create a copy of data with match probabilities
    X_data_with_probs = X_data.copy()
    X_data_with_probs['match_prob'] = match_probs
    
    # Create user profile dictionary for filtering
    user_profile_dict = user_profile.to_dict('records')[0]
    
    # Age range filtering
    age_min = user_profile_dict.get('age_o') - 5  # Allow some flexibility
    age_max = user_profile_dict.get('age_o') + 5
    age_filter = (X_data_with_probs['age'] >= age_min) & (X_data_with_probs['age'] <= age_max)
    
    # Gender filtering
    if user_profile_dict['gender'] == 0:  # If user is male
        gender_filter = X_data_with_probs['gender'] == 1  # Find females
    else:
        gender_filter = X_data_with_probs['gender'] == 0  # Find males
    
    # Additional preference filtering
    ambition_filter = np.abs(X_data_with_probs['ambition'] - user_profile_dict['ambition_partner']) <= 2
    attractive_filter = np.abs(X_data_with_probs['attractive_o'] - user_profile_dict['pref_attractive']) <= 2
    sincere_filter = np.abs(X_data_with_probs['sinsere_o'] - user_profile_dict['pref_sincere']) <= 2
    funny_filter = np.abs(X_data_with_probs['funny_o'] - user_profile_dict['pref_funny']) <= 2
    
    # Combine filters
    filtered_matches = X_data_with_probs[
        age_filter & 
        gender_filter & 
        ambition_filter & 
        attractive_filter & 
        sincere_filter & 
        funny_filter
    ]
    
    # Rank potential matches by match probability
    ranked_matches = filtered_matches.sort_values(by='match_prob', ascending=False)
    
    if len(ranked_matches) > 0:
        top_match = ranked_matches.head(1)
        match_prob = top_match['match_prob'].values[0]
        
        # Convert encoded values back to original
        race_list = ['Asian', 'Black', 'Caucasian', 'Hispanic', 'Other']
        top_match['race'] = race_list[int(top_match['race'].values[0])]
        top_match['gender'] = 'Female' if top_match['gender'].values[0] == 1 else 'Male'
        
        return top_match, match_prob
    else:
        return None, 0

# Load and process data
data = load_and_preprocess_data()
X = data.drop(columns=['match'])  # Features
y = data['match']  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Calculate model accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

# Streamlit UI
st.title("Matchmaker ")
st.write("Enter your preferences and find your match!")

# Collect user input
age = st.number_input("Your Age", min_value=18, max_value=100, step=1, value=25)
partner_age = st.slider("Preferred Partner Age Range", min_value=18, max_value=100, value=(25, 35))
gender = st.selectbox("Your Gender", ["male", "female"])
race = st.selectbox("Your Race", ["Asian", "Black", "Caucasian", "Hispanic", "Other"])

# Add ambition, intelligence, and other features
ambition = st.slider("Your Ambition Level", min_value=1, max_value=10, value=5)
ambition_partner = st.slider("Preferred Partner's Ambition Level", min_value=1, max_value=10, value=5)

# New sliders for preferred partner traits
pref_attractive = st.slider("Preferred Partner's Attractiveness", min_value=1, max_value=10, value=5)
pref_sincere = st.slider("Preferred Partner's Sincerity", min_value=1, max_value=10, value=5)
pref_funny = st.slider("Preferred Partner's Sense of Humor", min_value=1, max_value=10, value=5)

# Create a user profile
user_profile = {
    'age': age,
    'age_o': (partner_age[0] + partner_age[1]) / 2,  # Average of the preferred partner age range
    'gender': 0 if gender == 'ale' else 1,
    'race': ['Asian', 'Black', 'Caucasian', 'Hispanic', 'Other'].index(race),
    'ambition': ambition,
    'ambition_partner': ambition_partner,
    'pref_attractive': pref_attractive,
    'pref_sincere': pref_sincere,
    'pref_funny': pref_funny,
}

# Prepare user profile for model
user_profile_df = pd.DataFrame([user_profile])

# Show model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Suggest match
with open(WORKFLOW_NAME, "r") as f:
    prompt = json.load(f)


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    urllib.request.urlretrieve(f"http://{SERVER}/view?{url_values}", filename)


def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    filenames = []
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # Previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        if "images" in node_output:
            for image in node_output["images"]:
                get_image(image["filename"], image["subfolder"], image["type"])
                filenames.append(image["filename"])
    return filenames[0]


# Streamlit matchmaking integration with image generation
if st.button("Find My Match!"):
    match, match_prob = suggest_single_match(
        user_profile=user_profile_df,
        model=model,
        X_columns=X.columns,
        scaler=scaler,
        X_data=X,
        y_data=y,
    )

    if match is not None:
        st.write("Your Top Match:")
        display_columns = ["age", "gender", "race", "ambition", "match_prob", "attractive_o", "sinsere_o", "funny_o"]
        match_display = match[display_columns].copy()
        match_display.columns = ["Age", "Gender", "Race", "Ambition", "Match Probability", "Attractiveness", "Sincerity", "Sense of Humor"]
        st.dataframe(match_display)
        st.write(f"Match Probability: {match_prob * 100:.2f}%")

        # Connect to ComfyUI server and generate image
        ws = websocket.WebSocket()
        ws.connect(f"ws://{SERVER}/ws?clientId={client_id}")

        # Prepare the text description for the image generation
        # gender = "Male" if match["gender"].values[0] == 0 else "Female"
        gender = match["gender"].values[0]
        st.write(match["gender"].values[0])
        race = match["race"].values[0]
        ambition = "high" if match["ambition"].values[0] > 5 else "moderate"
        attractiveness = "attractive" if match["attractive_o"].values[0] > 5 else "average looking"
        sense_of_humor = "funny" if match["funny_o"].values[0] > 5 else "mild humor"

        text = f"A {gender} {race} with {ambition} ambition, {attractiveness}, with {sense_of_humor}."
        prompt["6"]["inputs"]["text"] = text
        prompt["3"]["inputs"]["seed"] = random.randint(0,1000)


        st.write(text)

        imgname = get_images(ws, prompt)
        ws.close()

        # Display the generated image
        st.image(imgname, caption="Generated Match Image")
    else:
        st.write("No matches found. Try adjusting your preferences.")