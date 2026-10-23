import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="SkinNova AI", page_icon="✚", layout="wide")

# Tuzha juna HTML/CSS frontend load karnyasathi (tuzhya fileche naav index.html kiwa templates/index.html asel te ithe taka)
html_path = "D:\Matlab Project\SkinDiseaseAI\frontend\index.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    
    # Render custom HTML/CSS/JS frontend inside Streamlit
    components.html(html_code, height=900, scrolling=True)
else:
    st.error(f"HTML file sapdat nahiye! Path check kara: {html_path}")