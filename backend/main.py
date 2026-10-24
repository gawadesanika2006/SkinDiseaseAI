from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
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
# FIND PROJECT FILES & PATHS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if "src" not in sys.path:
    sys.path.append("src")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from models.skin_model import create_model
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms

# ==========================================
# FASTAPI APP SETUP & CORS
# ==========================================
app = FastAPI(
    title="SkinNova AI",
    description="AI-based Skin Lesion classification prototype",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Convert class numbers to disease names
reverse_label_map = {
    value: key
    for key, value in LABEL_MAP.items()
}

# ==========================================
# CLINICAL KNOWLEDGE BASE & RISK METRICS
# ==========================================
DEFAULT_DISEASE_INFO = {
    "Nevus": {
        "risk_level": "Low Risk (Benign)",
        "risk_category": "benign",
        "badge_color": "#10b981",
        "description": "A nevus, commonly called a mole, is a very common benign skin growth formed by clusters of melanocytes (pigment-producing cells).",
        "causes": [
            "Genetic predisposition and inherited traits",
            "Normal developmental proliferation of melanocytes",
            "Ultraviolet (UV) sun exposure influencing mole count and pigmentation"
        ],
        "general_care": [
            "Apply broad-spectrum sunscreen (SPF 30+) daily",
            "Perform regular self-examinations following the ABCDE criteria",
            "Avoid picking, scratching, or rubbing the lesion"
        ],
        "medical_attention": "A changing, itching, bleeding, asymmetrical, or multi-colored mole should be examined promptly by a dermatologist."
    },
    "Melanoma, NOS": {
        "risk_level": "Critical Risk (Malignant)",
        "risk_category": "critical",
        "badge_color": "#ef4444",
        "description": "Melanoma is an aggressive form of skin cancer arising from melanocytes. Early clinical detection and intervention are essential for optimal outcomes.",
        "causes": [
            "Cumulative and intense ultraviolet (UV) radiation from sunlight or tanning beds",
            "History of severe, blistering sunburns, particularly during childhood",
            "Presence of numerous or dysplastic (atypical) nevi, or family history"
        ],
        "general_care": [
            "Strict sun avoidance, especially during peak UV hours (10 AM - 4 PM)",
            "Wear UV-protective clothing, wide-brimmed hats, and sunglasses",
            "Conduct monthly full-body skin self-checks"
        ],
        "medical_attention": "URGENT: Any suspected melanoma requires immediate clinical examination, dermoscopy, and an urgent biopsy by a dermatologist or oncologist."
    },
    "Basal cell carcinoma": {
        "risk_level": "High Risk (Malignant)",
        "risk_category": "urgent",
        "badge_color": "#f97316",
        "description": "Basal cell carcinoma (BCC) is the most frequent form of skin cancer. It typically grows slowly and rarely metastasizes, but can cause localized tissue destruction if untreated.",
        "causes": [
            "Chronic, cumulative ultraviolet (UV) solar radiation exposure",
            "Fair skin phototype, light hair, or light eye color",
            "Immune suppression or past therapeutic radiation"
        ],
        "general_care": [
            "Wear wide-brimmed hats and UV-blocking clothing outdoors",
            "Apply water-resistant high-SPF sun protection every 2 hours",
            "Promptly report persistent, non-healing skin sores"
        ],
        "medical_attention": "A sore that fails to heal after 3-4 weeks, repeatedly bleeds, or a pearly translucent bump should be evaluated for definitive dermatologic excision."
    },
    "Squamous cell carcinoma, NOS": {
        "risk_level": "High Risk (Malignant)",
        "risk_category": "urgent",
        "badge_color": "#f97316",
        "description": "Squamous cell carcinoma (SCC) is the second most common form of skin cancer, arising from epidermal keratinocytes. It may present as a firm red nodule or rough scaly patch.",
        "causes": [
            "Cumulative long-term lifetime UV exposure",
            "Evolution from untreated precancerous actinic keratoses",
            "Immunosuppression or exposure to chemical carcinogens"
        ],
        "general_care": [
            "Adhere strictly to sun protection protocols",
            "Avoid tanning salons and artificial UV sources completely",
            "Keep the affected skin clean, moisturized, and protected"
        ],
        "medical_attention": "A persistent, growing, crusted, or easily bleeding lesion requires prompt clinical assessment and dermatological biopsy."
    },
    "Solar or actinic keratosis": {
        "risk_level": "Moderate Risk (Precancerous)",
        "risk_category": "warning",
        "badge_color": "#f59e0b",
        "description": "Actinic keratosis is a rough, scaly precancerous skin lesion resulting from decades of UV damage. Some lesions may evolve into invasive squamous cell carcinoma if left untreated.",
        "causes": [
            "Long-term cumulative sun damage",
            "Repeated episodes of sunburn and UV radiation",
            "Indoor tanning bed exposure"
        ],
        "general_care": [
            "Daily use of broad-spectrum SPF 50+ sunscreen",
            "Protective clothing and minimizing direct sun exposure",
            "Gentle moisturization of dry, scaly areas"
        ],
        "medical_attention": "Actinic keratoses should be evaluated and managed by a dermatologist using cryotherapy, topical treatments, or photodynamic therapy to prevent progression."
    },
    "Pigmented benign keratosis": {
        "risk_level": "Low Risk (Benign)",
        "risk_category": "benign",
        "badge_color": "#10b981",
        "description": "A pigmented benign keratosis (such as a seborrheic keratosis or solar lentigo) is a harmless, non-cancerous skin growth that often has a 'stuck-on' appearance.",
        "causes": [
            "Natural skin aging and cellular turnover",
            "Genetic factors and family history",
            "Accumulated solar exposure over the lifespan"
        ],
        "general_care": [
            "Protect skin from excessive sun exposure",
            "Avoid scratching or picking at the lesion to prevent secondary irritation",
            "Moisturize skin with gentle emollients"
        ],
        "medical_attention": "Although benign, lesions that become irritated, inflamed, bleed, or mimic malignant melanoma should be professionally inspected by a dermatologist."
    },
    "Dermatofibroma": {
        "risk_level": "Low Risk (Benign)",
        "risk_category": "benign",
        "badge_color": "#10b981",
        "description": "A dermatofibroma is a harmless, firm, benign fibrous nodule in the dermis, commonly occurring on the lower legs and displaying a characteristic dimpling sign when pinched.",
        "causes": [
            "Exact etiology unknown; often considered a reactive fibrous proliferation",
            "Frequently develops after minor skin trauma, shaving nicks, or insect bites"
        ],
        "general_care": [
            "No medical intervention is generally needed for asymptomatic lesions",
            "Avoid aggressive scrubbing or irritation from clothing or shaving",
            "Monitor periodically for changes in appearance"
        ],
        "medical_attention": "Seek medical evaluation if the lesion changes significantly, grows rapidly, bleeds, or causes notable pain."
    }
}

# ==========================================
# LAZY LOAD AI MODEL (Optimized for Render Free Tier)
# ==========================================
device = torch.device("cpu")
model = None

def get_model():
    global model
    if model is None:
        print("Loading PyTorch model into memory...")
        m = create_model(num_classes=len(LABEL_MAP), pretrained=False)
        
        # Model path check - search common locations
        possible_paths = [
            "skin_disease_model.pth",
            os.path.join(project_root, "skin_disease_model.pth"),
            "backend/skin_disease_model.pth",
            os.path.join(current_dir, "skin_disease_model.pth"),
        ]
        model_path = next((p for p in possible_paths if os.path.exists(p)), "skin_disease_model.pth")
        print(f"Loading weights from: {model_path}")

        m.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        m.eval()
        
        # --- RENDER 512MB RAM OPTIMIZATION: Dynamic Quantization ---
        try:
            m = torch.quantization.quantize_dynamic(
                m, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Model successfully quantized for low memory!")
        except Exception as q_err:
            print(f"Quantization skipped: {q_err}")
            
        model = m
    return model


# ==========================================
# CONNECT TO MONGODB (Supports Local & Render Cloud)
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
disease_collection = None

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client["skin_disease_ai"]
    disease_collection = db["diseases"]
except Exception as e:
    print(f"MongoDB Note: {e} - Falling back to built-in clinical database.")


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SkinNova AI",
        "model_loaded": model is not None
    }


@app.get("/api/encyclopedia")
def get_encyclopedia():
    return {
        "status": "success",
        "diseases": DEFAULT_DISEASE_INFO,
        "classes": list(LABEL_MAP.keys())
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Get model lazily on request
        model_instance = get_model()

        # Read uploaded image
        image_bytes = await file.read()

        # Open and automatically handle EXIF orientation from phone cameras
        raw_image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

        # Preprocess image
        image_tensor = image_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)

        # AI prediction
        with torch.no_grad():
            output = model_instance(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Get top predicted disease name
        predicted_label = reverse_label_map[predicted_class.item()]
        confidence_percentage = confidence.item() * 100

        # Calculate Top-3 differential probabilities
        topk = min(3, len(LABEL_MAP))
        top_probs, top_indices = torch.topk(probabilities, topk, dim=1)
        top3 = [
            {
                "label": reverse_label_map[idx.item()],
                "prob": round(prob.item() * 100, 1),
                "risk_category": DEFAULT_DISEASE_INFO.get(reverse_label_map[idx.item()], {}).get("risk_category", "benign")
            }
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

        # Retrieve disease clinical guidance (Mongo or Built-in Fallback)
        fallback = DEFAULT_DISEASE_INFO.get(predicted_label, {})
        disease_info = None

        if disease_collection is not None:
            try:
                disease_info = disease_collection.find_one(
                    {"name": predicted_label},
                    {"_id": 0}
                )
            except Exception as e:
                print(f"Database query error: {e}")

        if not disease_info:
            disease_info = fallback

        description = disease_info.get("description", fallback.get("description", ""))
        causes = disease_info.get("causes", fallback.get("causes", []))
        general_care = disease_info.get("general_care", fallback.get("general_care", []))
        medical_attention = disease_info.get(
            "medical_attention",
            fallback.get("medical_attention", "Please consult a dermatologist for professional clinical evaluation.")
        )
        risk_level = fallback.get("risk_level", "Clinical Review Recommended")
        risk_category = fallback.get("risk_category", "warning")
        badge_color = fallback.get("badge_color", "#f59e0b")

        return {
            "prediction": predicted_label,
            "predicted_label": predicted_label,
            "confidence": round(confidence_percentage, 1),
            "risk_level": risk_level,
            "risk_category": risk_category,
            "badge_color": badge_color,
            "top3": top3,
            "description": description,
            "causes": causes,
            "general_care": general_care,
            "care": general_care,
            "medical_attention": medical_attention,
            "medical": medical_attention
        }

    finally:
        # Clear unused memory after prediction to keep memory low on Render
        gc.collect()


# ==========================================
# SERVE FRONTEND STATIC FILES
# ==========================================
frontend_dir = "frontend"
if not os.path.exists(frontend_dir):
    frontend_dir = os.path.join(project_root, "frontend")

if os.path.exists(frontend_dir):
    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True
        ),
        name="frontend"
    )
