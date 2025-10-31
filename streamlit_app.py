import streamlit as st

# Page Setup
homepage = st.Page(
    page="pages/homepage.py",
    title="Welcome to the Project",
    icon="👋",
    default=True
)

detection_app_page = st.Page(
    page="pages/detection_app.py", 
    title="Fire Smoke Segmentation Application",
    icon="🤖",
)


# Navigation Setup
pg = st.navigation(
    {
        "Home": [homepage],
        "Projects": [detection_app_page],
    }
)

# Share on All Pages
st.logo("static/asset/fire_smoke_icon.png", size="large")
st.sidebar.text("Created by 🎧 Kanon14")

# Run Application
pg.run()