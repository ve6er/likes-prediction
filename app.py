# streamlit_app.py
import streamlit as st
import os
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError
import json
import httpx

client = httpx.Client(
    headers={
        "x-ig-app-id": "936619743392459",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
    }
)

def get_followers(username: str):
    """Scrape Instagram user's data"""
    result = client.get(f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}")
    if result.status_code == 404:
        return None
    data = json.loads(result.content)
    if not data or not data.get("data") or not data["data"].get("user"):
        return None
    followers = data["data"]["user"]["edge_followed_by"]["count"]
    return followers

# Define constants
CATEGORIES = ["Cooking", "Fitness", "Fashion", "Photography"]
IMG_SIZE = (224, 224)

# Define functions
def load_and_prepare_image(image_file):
    try:
        img = load_img(image_file, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        return np.expand_dims(img_array, axis=0)
    except UnidentifiedImageError:
        st.warning("Uploaded file is not a valid image.")
        return None

def find_closest_rank(target_value, score_dict, followers):
    closest_rank = min(score_dict.keys(), key=lambda x: abs(x - target_value))
    corresponding_score = score_dict[closest_rank]
    predicted_likes = followers * corresponding_score
    return closest_rank, corresponding_score, predicted_likes

def test_model(model, image_array, csv_path, followers):
    prediction = model.predict(image_array)[0]
    df = pd.read_csv(csv_path)
    score_dict = dict(zip(df['Rank_Normalized_Score'], df['Score']))
    closest_rank, corresponding_score, predicted_likes = find_closest_rank(prediction, score_dict, followers)
    return closest_rank, corresponding_score, predicted_likes

# Streamlit App Interface
st.title("Image Rank and Likes Prediction App")
st.write("Select a category, upload an image, and enter your Instagram username to get predictions.")

# Category selection
category = st.selectbox("Select Category", CATEGORIES)

# File uploader for a single image
image_file = st.file_uploader("Upload an image (max 3MB)", type=["jpg", "jpeg", "png"])
if image_file and image_file.size > 3 * 1024 * 1024:
    st.warning("Image file size exceeds 3MB limit.")
    image_file = None
elif image_file:
    # Display the uploaded image
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

# Username input for follower count retrieval
username = st.text_input("Enter your Instagram username")

# Predict button
if st.button("Predict") and image_file and username:
    # followers = get_followers(username)
    followers=100000
    if followers is None:
        st.error(f"Username '{username}' not found.")
    else:
        # Load model based on selected category
        model_path = f"best_{category.lower()}_model.h5"
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            image_array = load_and_prepare_image(image_file)
            
            if image_array is not None:
                # Run prediction
                csv_path = f"csvs/{category}.csv"  # Assuming a CSV per category with this naming format
                if os.path.exists(csv_path):
                    closest_rank, corresponding_score, predicted_likes = test_model(model, image_array, csv_path, followers)
                    
                    # Display results
                    st.write(f"**Corresponding Score:** {corresponding_score}")
                    st.write(f"**Followers:** {followers}")
                    st.write(f"**Predicted Likes:** {predicted_likes}")
                    # closest_rank=90
                    st.markdown(f"""
                        <div style="width: 100%; background-color: #e0e0e0; padding: 8px; border-radius: 8px; position: relative;">
                            <div style="width: {closest_rank}%; background: linear-gradient(to right, red 0%, yellow {50*100/closest_rank}%, green {100*100/closest_rank}%); background-size: 300% 100%; background-position: 0 0; height: 20px; border-radius: 8px;"></div>
                        </div>
                        <div style="text-align: center; font-weight: bold; margin-top: 5px;">Closest Rank Normalized Score: {closest_rank:.2f}</div>
                    """, unsafe_allow_html=True)

                else:
                    st.error(f"Metadata CSV file '{csv_path}' not found for category '{category}'.")
        else:
            st.error(f"Model file '{model_path}' not found for category '{category}'.")