import streamlit as st
from PIL import Image
import torch
import os
import sys
from pymongo import MongoClient

sys.path.append("src")
from models.skin_model import create_model
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms

# Page Configuration
st.set_page_config(page_title="SkinNova AI", page_icon="✚", layout="centered")

# Custom CSS to match your original UI design
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #135d8a;
    }
    .upload-box {
        border: 2px dashed #4b5563;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("## ✚ SkinNova AI | Smart Skin Analysis")
st.markdown("Understand your skin better using Digital Image Processing and Deep Learning techniques.")
st.success("🟢 AI System Ready (1GB RAM Optimized)")

@st.cache_resource
def load_ai_model():
    device = torch.device("cpu")
    torch.set_num_threads(1)
    
    m = create_model(num_classes=len(LABEL_MAP))
    model_path = "skin_disease_model.pth"
    if not os.path.exists(model_path) and os.path.exists("backend/skin_disease_model.pth"):
        model_path = "backend/skin_disease_model.pth"
        
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    return m

with st.spinner("Loading AI Model into memory..."):
    model = load_ai_model()

reverse_label_map = {value: key for key, value in LABEL_MAP.items()}

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI)
    db = client["skin_disease_ai"]
    disease_collection = db["diseases"]
except Exception as e:
    disease_collection = None

st.markdown("---")
st.subheader("Upload Skin Image")
st.markdown("Select a clear image from your gallery or capture using your camera.")

# File Uploader (Supports both Gallery upload and Camera capture on mobile/desktop)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Selected Skin Image", use_column_width=True)
    
    if st.button("✦ Analyze Image"):
        with st.spinner("AI is analyzing your image..."):
            image_tensor = image_transforms(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_label = reverse_label_map[predicted_class.item()]
            confidence_percentage = round(confidence.item() * 100, 2)
            
            disease_info = None
            if disease_collection is not None:
                try:
                    disease_info = disease_collection.find_one({"name": predicted_label}, {"_id": 0})
                except:
                    pass
            
            st.markdown("---")
            st.subheader("📊 Analysis Result")
            st.metric(label="Prediction", value=predicted_label)
            st.metric(label="Confidence Score", value=f"{confidence_percentage}%")
            
            if disease_info:
                st.markdown(f"**Description:** {disease_info.get('description', '')}")
                causes = disease_info.get('causes', [])
                if causes:
                    st.markdown("**Causes / Associations:**")
                    for c in causes:
                        st.markdown(f"- {c}")
                care = disease_info.get('general_care', [])
                if care:
                    st.markdown("**General Care:**")
                    for cr in care:
                        st.markdown(f"- {cr}")
                st.markdown(f"**Medical Attention:** {disease_info.get('medical_attention', '')}")
            else:
                st.info("Please consult a dermatologist for professional evaluation.")