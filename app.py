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
st.set_page_config(page_title="Dermalens | AI Scanner", page_icon="✚", layout="wide")

# Custom CSS matching Dermalens Professional Light/Mint Design
st.markdown("""
    <style>
    .stApp {
        background-color: #f8faf9;
        color: #1f2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Navbar styling */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 30px;
    }
    .logo {
        font-weight: bold;
        font-size: 20px;
        color: #064e3b;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Main Header */
    .main-header {
        font-size: 34px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }
    .sub-header {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #065f46;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 10px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #047857;
    }

    /* Metrics & Results */
    .metric-container {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Top Navbar Simulation
st.markdown("""
    <div class="navbar">
        <div class="logo">
            <span style="background: #065f46; color: white; padding: 4px 8px; border-radius: 4px;">d</span> dermalens
        </div>
        <div style="color: #4b5563; font-size: 14px; display: flex; gap: 20px;">
            <span>Home</span>
            <span style="color: #065f46; font-weight: bold;">AI Scanner</span>
            <span>Encyclopedia</span>
            <span>Scan History</span>
            <span>Find Care</span>
            <span>About AI</span>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Realtime lesion screening & diagnostic lab.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Align your camera over the skin lesion for live AI detection or upload a clear dermoscopy photo.</div>', unsafe_allow_html=True)

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

# Two Column Layout (Step 01 & Step 02) matching Dermalens
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### STEP 01")
    st.markdown("#### Scan lesion")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Selected Skin Image", use_column_width=True)
    
    analysis_clicked = st.button("✦ Analyze Image")
    
    st.markdown("---")
    st.markdown("**Tips for Highest AI Accuracy:**")
    st.markdown("- Center the mole squarely inside the target reticle.")
    st.markdown("- Use bright, diffuse lighting (avoid direct camera flash glare).")
    st.markdown("- Keep surrounding skin free from shadows or hair where possible.")

with col2:
    st.markdown("### STEP 02")
    st.markdown("#### Diagnostic summary")
    st.markdown("---")
    
    if uploaded_file is not None and analysis_clicked:
        with st.spinner("AI is analyzing your skin image..."):
            image_tensor = image_transforms(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Get top-3 predictions
                top3_prob, top3_indices = torch.topk(probabilities, min(3, probabilities.size(1)))
            
            predicted_class = top3_indices[0][0].item()
            predicted_label = reverse_label_map[predicted_class]
            confidence_percentage = round(top3_prob[0][0].item() * 100, 2)
            
            disease_info = None
            if disease_collection is not None:
                try:
                    disease_info = disease_collection.find_one({"name": predicted_label}, {"_id": 0})
                except:
                    pass
            
            st.markdown(f"""
                <div class="metric-container">
                    <span style="color: #047857; font-size: 12px; font-weight: bold;">SCREENING COMPLETE</span>
                    <h2 style="color: #065f46; margin: 5px 0;">{predicted_label}</h2>
                    <p style="margin: 0; color: #374151;">Confidence: <b>{confidence_percentage}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**DIFFERENTIAL PROBABILITIES (TOP-3):**")
            for i in range(top3_prob.size(1)):
                idx = top3_indices[0][i].item()
                lbl = reverse_label_map[idx]
                prob = round(top3_prob[0][i].item() * 100, 2)
                st.markdown(f"- **{lbl}**: {prob}%")
                st.progress(int(prob))
            
            if disease_info:
                st.markdown("---")
                st.markdown("**Description:**")
                st.markdown(disease_info.get('description', 'No description available.'))
                
                causes = disease_info.get('causes', [])
                if causes:
                    st.markdown("**Associations & Risk Factors:**")
                    for c in causes:
                        st.markdown(f"- {c}")
                
                care = disease_info.get('general_care', [])
                if care:
                    st.markdown("**General Care Instructions:**")
                    for cr in care:
                        st.markdown(f"- {cr}")
                
                med = disease_info.get('medical_attention', '')
                if med:
                    st.markdown("**When to seek medical attention:**")
                    st.warning(med)
            else:
                st.info("Please consult a dermatologist for professional evaluation.")
    else:
        st.markdown("""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 20px; border-radius: 8px; text-align: center; color: #065f46;">
                <p style="margin: 0; font-weight: 600;">🟢 AWAITING IMAGE SCAN</p>
                <p style="margin: 5px 0 0 0; color: #4b5563;">Your result will appear here</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("**Description**")
        st.markdown("A screening summary will be available after lesion analysis.")