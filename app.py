import html
import io
import os
import re
import sys
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models

sys.path.append("src")
from models.label_map import LABEL_MAP  # noqa: E402
from preprocessing.transforms import image_transforms  # noqa: E402

st.set_page_config(
    page_title="SkinNova | AI Skin Scanner",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# DISEASE KNOWLEDGE BASE (built in, so the app works without MongoDB)
# ----------------------------------------------------------------------
DISEASES = {
    "Nevus": {
        "description": "A nevus, commonly called a mole, is a common skin growth.",
        "causes": ["Genetic factors", "Normal development of pigment-producing cells",
                   "Sun exposure can influence the number and appearance of moles"],
        "general_care": ["Protect the skin from excessive sun exposure",
                         "Monitor moles for changes in size, shape, or color",
                         "Avoid picking or irritating a mole"],
        "medical_attention": "A changing, itching, bleeding, or unusual mole should be examined by a dermatologist.",
    },
    "Melanoma, NOS": {
        "description": "Melanoma is a type of skin cancer involving pigment-producing cells.",
        "causes": ["UV radiation exposure", "Previous sun damage", "Genetic and other risk factors"],
        "general_care": ["Protect skin from UV exposure", "Avoid tanning beds",
                         "Monitor skin for changing or unusual spots"],
        "medical_attention": "A suspicious or changing pigmented lesion should be evaluated promptly by a dermatologist.",
    },
    "Basal cell carcinoma": {
        "description": "Basal cell carcinoma is a common type of skin cancer that usually grows slowly.",
        "causes": ["Long-term UV exposure", "Sun damage", "Indoor tanning exposure"],
        "general_care": ["Protect skin from excessive UV exposure", "Use sun protection",
                         "Have suspicious or persistent skin growths examined"],
        "medical_attention": "A sore that does not heal, repeatedly bleeds, or a suspicious new growth should be examined by a dermatologist.",
    },
    "Squamous cell carcinoma, NOS": {
        "description": "Squamous cell carcinoma is a type of skin cancer that develops from squamous cells.",
        "causes": ["UV radiation exposure", "Cumulative sun damage",
                   "Previous skin damage and certain risk factors"],
        "general_care": ["Protect skin from UV exposure", "Avoid tanning beds",
                         "Monitor persistent or changing skin lesions"],
        "medical_attention": "A persistent, growing, bleeding, crusting, or non-healing lesion should be evaluated by a dermatologist.",
    },
    "Solar or actinic keratosis": {
        "description": "Actinic keratosis is a precancerous skin growth associated with long-term UV damage.",
        "causes": ["Long-term sun exposure", "UV radiation", "Indoor tanning exposure"],
        "general_care": ["Protect the skin from UV exposure", "Use sun protection",
                         "Monitor rough or changing skin spots"],
        "medical_attention": "Actinic keratoses should be evaluated by a healthcare professional because some can progress to squamous cell carcinoma.",
    },
    "Pigmented benign keratosis": {
        "description": "A pigmented benign keratosis is a non-cancerous pigmented skin lesion.",
        "causes": ["Age-related skin changes", "Genetic factors", "Other factors affecting skin growth"],
        "general_care": ["Protect skin from excessive sun exposure",
                         "Avoid scratching or picking the lesion", "Monitor for noticeable changes"],
        "medical_attention": "A lesion that changes significantly, bleeds, or becomes symptomatic should be examined by a dermatologist.",
    },
    "Dermatofibroma": {
        "description": "A dermatofibroma is a common benign skin growth.",
        "causes": ["The exact cause is not always known",
                   "Some may develop after minor skin injury or insect bites"],
        "general_care": ["Avoid repeatedly irritating or scratching the lesion",
                         "Monitor for significant changes"],
        "medical_attention": "A rapidly changing, painful, bleeding, or unusual lesion should be evaluated by a dermatologist.",
    },
}
DEFAULT_ATTENTION = "Please consult a dermatologist for professional evaluation."

# ----------------------------------------------------------------------
# MODEL + OPTIONAL MONGODB
# ----------------------------------------------------------------------
reverse_label_map = {v: k for k, v in LABEL_MAP.items()}


@st.cache_resource(show_spinner="Loading AI model…")
def load_ai_model():
    torch.set_num_threads(1)
    net = models.resnet18(weights=None)  # same architecture as src/models/skin_model.py
    net.fc = nn.Linear(net.fc.in_features, len(LABEL_MAP))
    path = "skin_disease_model.pth"
    if not os.path.exists(path) and os.path.exists("backend/skin_disease_model.pth"):
        path = "backend/skin_disease_model.pth"
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


@st.cache_resource
def get_disease_collection():
    """Use MongoDB only if MONGO_URI is configured; otherwise built-in data is used."""
    uri = os.getenv("MONGO_URI")
    if not uri:
        try:
            uri = st.secrets.get("MONGO_URI")
        except Exception:
            uri = None
    if not uri:
        return None
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return client["skin_disease_ai"]["diseases"]
    except Exception:
        return None


def get_info(name):
    info = None
    coll = get_disease_collection()
    if coll is not None:
        try:
            info = coll.find_one({"name": name}, {"_id": 0})
        except Exception:
            info = None
    return info or DISEASES.get(name, {})


def analyze(image: Image.Image):
    model = load_ai_model()
    tensor = image_transforms(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_p, top_i = torch.topk(probs, 3)
    top3 = [{"label": reverse_label_map[i.item()], "prob": round(p.item() * 100, 2)}
            for p, i in zip(top_p, top_i)]
    best = top3[0]["label"]
    info = get_info(best)
    return {
        "label": best,
        "confidence": top3[0]["prob"],
        "top3": top3,
        "description": info.get("description", ""),
        "causes": info.get("causes", []),
        "care": info.get("general_care", []),
        "medical": info.get("medical_attention") or DEFAULT_ATTENTION,
    }


# ----------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&display=swap');
:root{--g:#14604f;--gd:#0d4538;--gl:#e3eee9;--bg:#eef3f0;--ink:#17302a;--mut:#5b6d66;--line:#dce5e0;--warn:#fbeeea;--warnb:#f0c9bf;--warnt:#a2392a}
html,body,.stApp,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,#f4f8f5 0%,#e9f0ec 100%)}
#MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding:1rem 1rem 4rem!important;max-width:980px}
.serif{font-family:'Newsreader',Georgia,serif}
/* brand bar */
.brand{display:flex;align-items:center;gap:12px;padding:6px 0 14px}
.brand .logo{width:40px;height:40px;border-radius:50%;background:var(--g);color:#fff;display:flex;align-items:center;justify-content:center;font-family:'Newsreader',serif;font-size:22px;font-weight:500}
.brand .name{font-size:19px;font-weight:600;letter-spacing:-.2px}
/* nav (radio as pills) */
div[role="radiogroup"]{display:flex;flex-wrap:nowrap;overflow-x:auto;gap:8px;padding-bottom:6px;scrollbar-width:none}
div[role="radiogroup"]::-webkit-scrollbar{display:none}
div[role="radiogroup"] label{background:#fff;border:1px solid var(--line);border-radius:999px;padding:7px 16px;white-space:nowrap;cursor:pointer;margin:0}
div[role="radiogroup"] label>div:first-child{display:none}
div[role="radiogroup"] label p{font-size:14px;font-weight:500;color:var(--mut)}
div[role="radiogroup"] label:has(input:checked){background:var(--g);border-color:var(--g)}
div[role="radiogroup"] label:has(input:checked) p{color:#fff}
/* hero */
.eyebrow{font-size:11px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;color:var(--g);margin:18px 0 8px}
.hero h1,.h1{font-family:'Newsreader',serif;font-weight:500;font-size:clamp(38px,9vw,64px);line-height:1.05;letter-spacing:-1px;margin:0 0 14px;color:var(--ink)}
.hero h1 em,.h1 em{color:var(--g);font-style:italic}
.lead{font-size:17px;line-height:1.65;color:var(--mut);max-width:620px;margin-bottom:18px}
.stats{display:flex;gap:12px;overflow-x:auto;padding:4px 2px 12px;scrollbar-width:none}
.stat{flex:0 0 auto;min-width:150px;display:flex;align-items:center;gap:12px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.stat b{font-family:'Newsreader',serif;font-size:38px;font-weight:500;color:var(--g);line-height:1}
.stat span{font-size:13px;font-weight:500;line-height:1.3;color:var(--ink)}
.rule{border:0;border-top:1px solid var(--line);margin:22px 0}
.h2{font-family:'Newsreader',serif;font-size:clamp(28px,6vw,40px);font-weight:500;margin:4px 0 14px}
.suite{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.suite div{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px}
.suite .ic{font-size:26px}
.suite h4{font-family:'Newsreader',serif;font-size:22px;font-weight:500;margin:8px 0 4px}
.suite p{font-size:14px;color:var(--mut);margin:0;line-height:1.5}
/* cards */
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line)!important;border-radius:14px!important;padding:6px 8px}
.step{display:flex;justify-content:space-between;font-size:11px;font-weight:700;letter-spacing:1.4px;color:var(--g);text-transform:uppercase}
.step span:last-child{color:var(--mut);font-weight:500}
.ctitle{font-family:'Newsreader',serif;font-size:28px;font-weight:500;margin:2px 0 12px}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:auto;padding:12px 14px;flex:1}
.stTabs [data-baseweb="tab"] p{font-size:14px;font-weight:500;color:var(--mut)}
.stTabs [aria-selected="true"]{background:var(--gl)}
.stTabs [aria-selected="true"] p{color:var(--g);font-weight:600}
.stTabs [data-baseweb="tab-highlight"]{background:var(--g)}
/* uploader / camera */
[data-testid="stFileUploaderDropzone"]{border:2px dashed #7fae9f;background:#fbfdfc;border-radius:10px;padding:26px 14px}
[data-testid="stFileUploaderDropzone"] button{border-radius:999px;border:1px solid var(--g);color:var(--g);background:#fff}
[data-testid="stCameraInput"] button{border-radius:999px}
/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:999px;font-weight:600;padding:.6rem 1.2rem;border:1px solid var(--line);background:#fff;color:var(--ink);width:100%}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:var(--g);border-color:var(--g);color:#fff}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{background:var(--gd);border-color:var(--gd);color:#fff}
.stButton>button:disabled{background:#a9b9b2!important;border-color:#a9b9b2!important;color:#fff!important}
/* result */
.res{background:var(--gl);border-left:4px solid var(--g);border-radius:4px 10px 10px 4px;padding:18px 18px 16px;margin:6px 0 14px}
.res .tag{font-size:10.5px;font-weight:700;letter-spacing:1.4px;color:var(--g);text-transform:uppercase}
.res .tag:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--g);margin-right:7px}
.res h3{font-family:'Newsreader',serif;font-size:32px;font-weight:500;line-height:1.1;margin:8px 0 8px}
.res .conf{font-size:14px;color:var(--mut)}
.pt{font-size:12px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;margin:16px 0 8px}
.pr{display:flex;justify-content:space-between;font-size:15px;font-weight:500;margin-top:10px}
.pr b{color:var(--g)}
.bar{height:6px;background:#dbe6e1;border-radius:4px;overflow:hidden;margin-top:4px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#14604f,#3f9a82);border-radius:4px}
.sub{font-size:13.5px;font-weight:700;margin:16px 0 4px}
.txt,.txt li{font-size:14.5px;color:var(--mut);line-height:1.55}
.txt ul{margin:0;padding-left:20px}
.warn{background:var(--warn);border:1px solid var(--warnb);border-radius:10px;padding:14px;margin-top:16px;display:flex;gap:12px}
.warn .i{width:26px;height:26px;border-radius:50%;background:#f6d8d0;color:var(--warnt);display:flex;align-items:center;justify-content:center;font-weight:700;flex:0 0 auto}
.warn b{font-size:14px}.warn p{margin:4px 0 0;font-size:14px;color:#6b5851;line-height:1.5}
.tips{background:var(--gl);border-radius:10px;padding:14px 16px;margin-top:14px;font-size:14px}
.tips ul{margin:6px 0 0;padding-left:18px;color:var(--mut)}
.note{font-size:12.5px;color:var(--mut);margin-top:8px}
.disc{font-size:12.5px;color:var(--mut);text-align:center;margin-top:34px;line-height:1.6}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line)!important;border-radius:12px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s: str) -> str:
    """Collapse indentation/newlines so Markdown never turns HTML into a code block."""
    return re.sub(r"\n\s*", "", s).strip()


def md(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


def ul(items):
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


# ----------------------------------------------------------------------
# STATE + NAV
# ----------------------------------------------------------------------
PAGES = ["Home", "AI Scanner", "Encyclopedia", "Scan History", "About AI"]
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("saved", False)


def go(page):
    st.session_state.page = page


md("""<div class="brand"><div class="logo">S</div><div class="name">SkinNova</div></div>""")
st.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed")
page = st.session_state.page


# ----------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------
def page_home():
    md("""
    <div class="hero">
      <div class="eyebrow">Dermatology AI Portal</div>
      <h1>Early detection starts with <em>closer observation.</em></h1>
      <p class="lead">SkinNova combines deep learning computer vision with clinical knowledge
      bases to deliver real-time skin lesion screening, mole tracking, and educational guidance.</p>
    </div>""")
    st.button("Launch Live AI Scanner 🔬", type="primary", on_click=go, args=("AI Scanner",), key="h1")
    st.button("Browse Disease Encyclopedia 📚", on_click=go, args=("Encyclopedia",), key="h2")
    md(f"""
    <div class="stats">
      <div class="stat"><b>{len(LABEL_MAP):02d}</b><span>Classified<br>Skin<br>Lesions</span></div>
      <div class="stat"><b>Top-3</b><span>Differential<br>Diagnoses</span></div>
      <div class="stat"><b>AI</b><span>ResNet-18<br>Deep Learning</span></div>
    </div>
    <hr class="rule">
    <div class="eyebrow" style="margin-top:0">Clinical Suite</div>
    <div class="h2">Explore the SkinNova Platform</div>
    <div class="suite">
      <div><div class="ic">📷</div><h4>AI Scanner</h4><p>Scan with your camera or upload a photo for instant lesion screening.</p></div>
      <div><div class="ic">📚</div><h4>Encyclopedia</h4><p>Learn about the seven lesion types SkinNova can recognise.</p></div>
      <div><div class="ic">🕘</div><h4>Scan History</h4><p>Keep track of the scans you saved during this session.</p></div>
      <div><div class="ic">🧠</div><h4>About AI</h4><p>How the model works and what its limits are.</p></div>
    </div>""")


def result_html(r):
    bars = "".join(
        f'<div class="pr"><span>{html.escape(t["label"])}</span><b>{t["prob"]}%</b></div>'
        f'<div class="bar"><i style="width:{min(t["prob"], 100)}%"></i></div>'
        for t in r["top3"]
    )
    return H(f"""
    <div class="res">
      <div class="tag">Screening complete</div>
      <h3>{html.escape(r["label"])}</h3>
      <div class="conf">Confidence: {r["confidence"]}%</div>
      <div class="pt">Differential probabilities (Top-3):</div>{bars}
    </div>
    <div class="sub">Description</div>
    <div class="txt">{html.escape(r["description"] or "No specific details available.")}</div>
    <div class="sub">Associations &amp; Risk Factors</div><div class="txt">{ul(r["causes"]) if r["causes"] else "Not available."}</div>
    <div class="sub">General Care Instructions</div><div class="txt">{ul(r["care"]) if r["care"] else "Not available."}</div>
    <div class="warn"><div class="i">!</div><div><b>When to seek medical attention</b>
    <p>{html.escape(r["medical"])}</p></div></div>""")


def summary_file(r):
    return f"""<!doctype html><meta charset="utf-8"><title>SkinNova Summary</title>
<body style="font-family:Georgia,serif;max-width:720px;margin:40px auto;color:#17302a">
<h1>SkinNova – Screening Summary</h1><p>{datetime.now():%d %b %Y, %H:%M}</p>
<h2>{html.escape(r['label'])}</h2><p>Confidence: {r['confidence']}%</p>
<h3>Differential probabilities</h3>{ul([f"{t['label']} – {t['prob']}%" for t in r['top3']])}
<h3>Description</h3><p>{html.escape(r['description'])}</p>
<h3>Associations &amp; Risk Factors</h3>{ul(r['causes'])}
<h3>General Care</h3>{ul(r['care'])}
<h3>When to seek medical attention</h3><p>{html.escape(r['medical'])}</p>
<hr><small>Screening aid only – not a medical diagnosis. Consult a dermatologist.</small></body>"""


def page_scanner():
    md("""
    <div class="eyebrow">Dermatology AI Scanner</div>
    <div class="h1" style="font-size:clamp(34px,8vw,56px)">Realtime lesion screening <em>&amp; diagnostic lab.</em></div>
    <p class="lead">Align your camera over the skin lesion for live AI detection or upload a clear dermoscopy photo.</p>""")

    c1, c2 = st.columns([1, 1.05], gap="medium")
    image, name = None, None

    with c1:
        with st.container(border=True):
            md("""<div class="step"><span>Step 01</span><span>01</span></div><div class="ctitle">Scan lesion</div>""")
            t_live, t_up = st.tabs(["📡 Live Realtime Scanner", "📁 Upload Image File"])
            with t_live:
                shot = st.camera_input("Take a photo of the lesion", label_visibility="collapsed")
                if shot is not None:
                    image, name = Image.open(shot), "camera-scan.jpg"
            with t_up:
                up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"],
                                      label_visibility="collapsed")
                if up is not None:
                    image, name = Image.open(up), up.name

            if image is not None:
                image = ImageOps.exif_transpose(image).convert("RGB")
                st.image(image, caption=f"{name} · Ready for screening", use_container_width=True)

            go_btn = st.button("Analyze image →", type="primary", disabled=image is None, key="analyze")
            md("""<div class="note">• Image is used for this analysis only</div>
            <div class="tips"><b>💡 Tips for Highest AI Accuracy:</b><ul>
            <li>Center the mole squarely inside the target reticle.</li>
            <li>Use bright, diffuse lighting (avoid direct camera flash glare).</li>
            <li>Keep surrounding skin free from shadows or hair where possible.</li></ul></div>""")

    if go_btn and image is not None:
        with st.spinner("Analyzing image with AI…"):
            res = analyze(image)
        thumb = image.copy()
        thumb.thumbnail((240, 240))
        res["thumb"] = thumb
        res["name"] = name
        st.session_state.result, st.session_state.saved = res, False

    with c2:
        with st.container(border=True):
            md("""<div class="step"><span>Step 02</span><span>02</span></div><div class="ctitle">Diagnostic summary</div>""")
            r = st.session_state.result
            if r is None:
                md("""
                <div class="res"><div class="tag">Analysis in progress</div>
                <h3>Your result will appear here</h3><div class="conf">Confidence: –</div></div>
                <div class="sub">Description</div><div class="txt">A screening summary will be available after lesion analysis.</div>
                <div class="sub">Associations &amp; Risk Factors</div><div class="txt"><ul><li>Not available yet</li></ul></div>
                <div class="sub">General Care Instructions</div><div class="txt"><ul><li>Not available yet</li></ul></div>
                <div class="warn"><div class="i">!</div><div><b>When to seek medical attention</b>
                <p>Consult a qualified healthcare professional or dermatologist about any concerning, changing, or persistent skin lesion.</p></div></div>""")
            else:
                st.markdown(result_html(r), unsafe_allow_html=True)
                if st.session_state.saved:
                    st.button("✓ Saved to History", disabled=True, key="saved_btn")
                elif st.button("💾 Save to My Scan History", type="primary", key="save"):
                    st.session_state.history.insert(0, {**r, "time": datetime.now()})
                    st.session_state.saved = True
                    st.rerun()
                st.download_button("📄 Print / Save Summary", summary_file(r), "skinnova-summary.html",
                                   "text/html", key="dl")
                if st.session_state.saved:
                    md("""<div class="note" style="color:#14604f;font-weight:600">✓ Scan saved to your History!</div>""")


def page_encyclopedia():
    md("""<div class="eyebrow">Disease Encyclopedia</div>
    <div class="h1" style="font-size:clamp(34px,8vw,52px)">Know the <em>lesions.</em></div>
    <p class="lead">Educational information about the seven lesion types SkinNova can classify.</p>""")
    for name in LABEL_MAP:
        d = get_info(name)
        with st.expander(name):
            md(f"""<div class="txt">{html.escape(d.get("description", ""))}</div>
            <div class="sub">Associations &amp; Risk Factors</div><div class="txt">{ul(d.get("causes", []))}</div>
            <div class="sub">General Care Instructions</div><div class="txt">{ul(d.get("general_care", []))}</div>
            <div class="warn"><div class="i">!</div><div><b>When to seek medical attention</b>
            <p>{html.escape(d.get("medical_attention", DEFAULT_ATTENTION))}</p></div></div>""")


def page_history():
    md("""<div class="eyebrow">Scan History</div>
    <div class="h1" style="font-size:clamp(34px,8vw,52px)">Your saved <em>scans.</em></div>""")
    hist = st.session_state.history
    if not hist:
        md("""<p class="lead">No scans saved yet. Analyze an image and tap <b>Save to My Scan History</b>.</p>""")
        st.button("Open AI Scanner", type="primary", on_click=go, args=("AI Scanner",), key="hs")
        return
    for h in hist:
        with st.container(border=True):
            a, b = st.columns([1, 2])
            a.image(h["thumb"], use_container_width=True)
            b.markdown(H(f"""<div class="step"><span>{h["time"]:%d %b %Y · %H:%M}</span></div>
            <div class="ctitle" style="font-size:24px;margin:2px 0">{html.escape(h["label"])}</div>
            <div class="conf" style="color:#5b6d66">Confidence: {h["confidence"]}%</div>"""),
                       unsafe_allow_html=True)
    if st.button("Clear history", key="clr"):
        st.session_state.history = []
        st.rerun()


def page_about():
    md("""<div class="eyebrow">About the AI</div>
    <div class="h1" style="font-size:clamp(34px,8vw,52px)">How SkinNova <em>works.</em></div>
    <div class="txt" style="font-size:16px;line-height:1.7;max-width:640px">
    <p>SkinNova uses a <b>ResNet-18</b> convolutional neural network fine-tuned to recognise seven skin lesion
    categories. Your photo is resized to 224×224, normalised, and passed through the network, which returns a
    probability for each category. The three most likely categories are shown as the differential.</p>
    <p>The educational text (description, risk factors, care and when to seek help) comes from a curated
    knowledge base.</p></div>
    <div class="warn"><div class="i">!</div><div><b>Important</b>
    <p>SkinNova is a screening and education aid, not a medical device. It cannot diagnose skin cancer.
    Always have any new, changing, bleeding or persistent lesion examined by a qualified dermatologist.</p></div></div>""")


{"Home": page_home, "AI Scanner": page_scanner, "Encyclopedia": page_encyclopedia,
 "Scan History": page_history, "About AI": page_about}[page]()

md("""<div class="disc">SkinNova is an AI screening aid and does not replace professional medical advice.</div>""")
