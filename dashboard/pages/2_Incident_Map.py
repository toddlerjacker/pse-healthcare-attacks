import streamlit as st
import streamlit.components.v1 as components
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import MAP_PATH

st.set_page_config(
    page_title="Incident Map",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container {
        padding: 1rem 1rem 0rem 1rem;
    }
    iframe {
        height: calc(100vh - 160px) !important;
        width: 100% !important;
    }
    /* Prevent the outer page from scrolling */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Incident Map")
st.markdown(
    "<p style='font-size:13px; color:#888888;'>"
    "Data: Insecurity Insight / SHCC  ·  Oct 2023–Feb 2026<br>"
    "Source data has 25km precision; exact incident locations unavailable. "
    "Points distributed within official PSE boundaries for visualization only.<br>"
    "Do not interpret dot positions as precise locations."
    "</p>",
    unsafe_allow_html=True
)

with open(MAP_PATH, "r", encoding="utf-8") as f:
    map_html = f.read()

components.html(map_html, height=800, scrolling=False)

st.caption(
    "Coordinates jittered within official HDX/OCHA PSE boundaries. "
    "Points represent approximate locations within a 25km precision zone — "
    "not exact incident coordinates."
)
