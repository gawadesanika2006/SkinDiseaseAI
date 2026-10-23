import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="SkinNova AI", page_icon="✚", layout="wide")

# Path to your original HTML frontend file
html_path = "frontend/index.html"

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    
    # Render your original HTML/CSS/JS frontend inside Streamlit wrapper
    components.html(html_code, height=900, scrolling=True)
else:
    st.error(f"HTML file sapdat nahiye! Path check kara: {html_path}")