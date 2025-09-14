import streamlit as st

# Page Setup


detection_app_page = st.Page(
    page="app_pages/detection_app.py", 
    title="Fire Smoke Segmentation Application",
    icon="🤖",
)


# Navigation Setup
pg = st.navigation(
    {
        "Projects": [detection_app_page],
    }
)

# Share on All Pages
st.logo("static/asset/fire_smoke_icon.png", size="large")
st.sidebar.text("Created by 🎧 Kanon14")

# Run Application
pg.run()