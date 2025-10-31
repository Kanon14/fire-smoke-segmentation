import streamlit as st

# Main Content
st.title("🔥 :orange[Fire & Smoke Instance Segmentation] 🔥")
st.header("Real-time detection and segmentation to enhance early fire response")
st.write("")

st.subheader("📜 About This Project")
st.write(
    """
    This **:orange[Fire & Smoke Instance Segmentation]** app uses modern **:orange[computer vision]** (YOLO-Seg)
    to detect and segment **:red[fire]** and **:gray[smoke]** in images and videos.  
    The goal is to support **early warnings**, improve **safety monitoring**, and assist **emergency response** teams.
    """
)

st.subheader("🎯 What it can do")
st.markdown(
    """
    - **Segment** fire/smoke regions with pixel-level masks  
    - **Label** detections with class names and confidences  
    - **Visualize** results instantly for uploaded media  
    """
)

st.info("💡 **Get started:** Use the sidebar to upload an image or video for segmentation.")
st.write("**Let’s build safer spaces with smarter vision! 🧯🌫️**")

# Tip: replace with your banner image if available
BANNER_PATH = "static/asset/fire_smoke_banner.jpg"  # e.g., place your image here
st.image(BANNER_PATH, caption="Fire & Smoke Instance Segmentation", width="stretch")
