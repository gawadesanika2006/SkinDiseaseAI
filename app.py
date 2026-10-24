import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import base64
import io
from PIL import Image
import torch
from pymongo import MongoClient

sys.path.append("src")
from models.skin_model import create_model
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms

st.set_page_config(page_title="SkinNova AI | AI Scanner", page_icon="✚", layout="wide")

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

model = load_ai_model()
reverse_label_map = {value: key for key, value in LABEL_MAP.items()}

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI)
    db = client["skin_disease_ai"]
    disease_collection = db["diseases"]
except Exception:
    disease_collection = None

html_path = "frontend/index.html"
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=920, scrolling=True)
else:
    st.error("HTML file not found in frontend/index.html")