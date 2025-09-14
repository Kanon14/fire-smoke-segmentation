import io
import cv2
import time
import sys
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image, ImageOps
from fireSmoke.exception import AppException
from fireSmoke.pipeline.training_pipeline import TrainPipeline


# Title of the application
st.title("🔥:orange[Fire Smoke Segmentation Application]🔥")

# Sidebar menu for app features
menu = st.sidebar.radio("Choose a feature:", ["Train Model", "Image Detection", "Webcam Detection", "IP Webcam Detection"])

# Model Training Pipeline
if menu == "Train Model":
    st.header("Train the Fire Smoke Segmentation Model", divider="green")
    if st.button("Start Training"):
        st.info("Training in progress. Please wait...")
        obj = TrainPipeline() # Instantiate the training pipeline object
        obj.run_pipeline()  # Run the training pipeline
        st.success("Model training and export completed successfully!")
        
# Image Detection
elif menu == "Image Detection":
    model = YOLO("yolo_seg_train/yolo12n-seg.pt")
    st.header("📱 Upload an Image for Fire Smoke Segmentation", divider="green")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image and results side by side
        col1, col2 = st.columns(2)

        # Convert uploaded file to PIL image and transpose for correct display
        pil_img = Image.open(io.BytesIO(uploaded_file.read()))
        pil_img = ImageOps.exif_transpose(pil_img)     
        pil_img = pil_img.convert("RGB") 

        with col1:
            st.image(pil_img, caption="Uploaded Image", width="stretch", channels="BGR")

       # Perform detection
        with st.spinner("Detecting objects..."):
            results = model.predict(pil_img, conf=0.25)[0]
            print(results)
            detections = sv.Detections.from_ultralytics(results)
            mask_annotator = sv.MaskAnnotator()
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

            annotated_image = pil_img.copy()
            annotated_image = mask_annotator.annotate(annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(annotated_image, detections=detections)
              
        with col2:
            st.image(annotated_image, caption="Detection Results", width="stretch", channels="BGR")
        st.success("Detection complete!")
    else:
        st.warning("⚠️ Please upload an image file to proceed.")