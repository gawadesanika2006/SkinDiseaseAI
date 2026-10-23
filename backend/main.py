from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import io
import sys
import os
import gc
from pymongo import MongoClient

# ==========================================
# RENDER 512MB RAM MEMORY OPTIMIZATION
# ==========================================
torch.set_num_threads(1)  # Limit CPU threads to save RAM

# ==========================================
# FIND PROJECT FILES
# ==========================================

sys.path.append("src")

from models.skin_model import create_model
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms


# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title="SkinNova AI",
    description="AI-based Skin Lesion classfication prototype",
    version="1.0.0"
)

# ==========================================
# LAZY LOAD AI MODEL (Optimized for Render Free Tier)
# ==========================================

device = torch.device("cpu")
model = None

def get_model():
    global model
    if model is None:
        print("Loading PyTorch model into memory...")
        m = create_model(num_classes=len(LABEL_MAP))
        
        # Model path check
        model_path = "skin_disease_model.pth"
        if not os.path.exists(model_path) and os.path.exists("backend/skin_disease_model.pth"):
            model_path = "backend/skin_disease_model.pth"

        m.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        m.eval()
        model = m
    return model

# Convert class numbers to disease names
reverse_label_map = {
    value: key
    for key, value in LABEL_MAP.items()
}


# ==========================================
# CONNECT TO MONGODB (Supports Local & Render Cloud)
# ==========================================

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

try:
    client = MongoClient(MONGO_URI)
    db = client["skin_disease_ai"]
    disease_collection = db["diseases"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")


# ==========================================
# PREDICT API
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        # Get model lazily on request
        model_instance = get_model()

        # Read uploaded image
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Preprocess image
        image_tensor = image_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)

        # AI prediction
        with torch.no_grad():
            output = model_instance(image_tensor)
            probabilities = torch.softmax(
                output,
                dim=1
            )
            confidence, predicted_class = torch.max(
                probabilities,
                1
            )

        # Get disease name
        predicted_label = reverse_label_map[
            predicted_class.item()
        ]

        confidence_percentage = (
            confidence.item() * 100
        )

        # Get information from MongoDB safely
        disease_info = None
        try:
            disease_info = disease_collection.find_one(
                {
                    "name": predicted_label
                },
                {
                    "_id": 0
                }
            )
        except Exception as e:
            print(f"Database query error: {e}")

        # Return result
        if disease_info:
            return {
                "prediction": predicted_label,
                "confidence": round(
                    confidence_percentage,
                    2
                ),
                "description": disease_info.get(
                    "description",
                    ""
                ),
                "causes": disease_info.get(
                    "causes",
                    []
                ),
                "general_care": disease_info.get(
                    "general_care",
                    []
                ),
                "medical_attention": disease_info.get(
                    "medical_attention",
                    ""
                )
            }

        # Fallback
        return {
            "prediction": predicted_label,
            "confidence": round(
                confidence_percentage,
                2
            ),
            "description": "",
            "causes": [],
            "general_care": [],
            "medical_attention":
                "Please consult a dermatologist "
                "for professional evaluation."
        }

    finally:
        # Clear unused memory after prediction
        gc.collect()


# ==========================================
# SERVE FRONTEND
# ==========================================

app.mount(
    "/",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)