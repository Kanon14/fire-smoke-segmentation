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
        
# Webcam Detection
elif menu == "Webcam Detection":
    model = YOLO("yolo_seg_train/yolo12n-seg.pt")
    st.header("🎥 Real-Time Detection from Webcam", divider="green")
    
    # Create two columns for Start and Stop Buttons
    col1, col2 = st.columns(2)
    start_button = col1.button("▶️ Start Webcam")
    stop_button = col2.button("⏹️ Stop Webcam")
    
    if start_button:
        video_placeholder = st.empty()  # Placeholder for displaying video frames
        fps_placeholder = st.empty()  # Placeholder for displaying FPS value

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Open default webcam
        cap.set(3, 1280)  # Set frame width
        cap.set(4, 720)   # Set frame height
        cap.set(cv2.CAP_PROP_FOURCC, 0x32595559) # CAP_PROP_FOURCC: 4-character code of codec
        cap.set(cv2.CAP_PROP_FPS, 30)            # CAP_PROP_FPS: Frame rate
        
        prev_frame_time = 0 # Previous frame time
        
        try:
            while cap.isOpened():
                if stop_button: # Check if Stop button is pressed
                    st.info("Webcam stopped")
                    break
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
            
                # Perform detection/tracking using YOLO model
                results =model.track(source=frame, verbose=False, device="cuda", stream=True, persist=True)
                for res in results:
                    annotated_frame = res.plot() # Annotate the frame with detection results
                    
                # Calculate FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
                prev_frame_time = new_frame_time
            
                # Update the video placeholder with the annotated frame
                video_placeholder.image(annotated_frame, width="stretch", channels="BGR")
                
                # Update the FPS placeholder with the current FPS value
                fps_placeholder.markdown(f"**FPS:** {int(fps)}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise AppException(e, sys)
        
        finally:
            cap.release() # Release webcam resource when done
            st.info("Webcam stopped")
            
# IP Webcam Detection
elif menu == "IP Webcam Detection":
    model = YOLO("yolo_seg_train/yolo12n-seg.pt")
    st.header("🧿 Real-Time Detection from IP Webcam", divider="green")

    # Create two columns for Start and Stop Buttons
    col1, col2 = st.columns(2)
    start_button = col1.button("▶️ Start IP Webcam")
    stop_button = col2.button("⏹️ Stop IP Webcam")

    ip_url = st.text_input("Enter IP Webcam URL (e.g., http://192.168.100.4:8080/video):")

    if start_button and ip_url:
        video_placeholder = st.empty()  # Placeholder for displaying video frames
        fps_placeholder = st.empty()  # Placeholder for displaying FPS value
        
        cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)  # Open IP webcam feed
        cap.set(3, 640)  # Reduce resolution to 640x480 for performance
        cap.set(4, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increase buffer size
        cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS
        
        prev_frame_time = 0 # Previous frame time

        try:
            while cap.isOpened():
                if stop_button:  # Check if Stop button is pressed
                    st.info("IP Webcam stopped")
                    break

                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read from IP webcam. Skipping frame...")
                    continue  # Skip bad frames instead of breaking the loop

                # Perform detection using YOLO model
                try:
                    results = model.track(source=frame, verbose=False, device="cuda", stream=True, persist=True)
                    for res in results:
                        annotated_frame = res.plot()  # Annotate the frame with detection results
                        
                    # Calculate FPS
                    new_frame_time = time.time()
                    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
                    prev_frame_time = new_frame_time   

                    # Update the video placeholder with the annotated frame
                    video_placeholder.image(annotated_frame, width="stretch", channels="BGR")
                    
                    # Update the FPS placeholder with the current FPS value
                    fps_placeholder.markdown(f"**FPS:** {int(fps)}")
                    
                except Exception as e:
                    st.warning(f"Error during YOLO detection: {str(e)}. Skipping this frame...")

                # Add delay for stability
                time.sleep(0.05)  # Add a 50 ms delay to reduce system load

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise AppException(e, sys)

        finally:
            cap.release()  # Release IP webcam resource when done
            st.info("IP Webcam stopped")

    elif start_button and not ip_url:
        st.error("Please enter a valid IP Webcam URL.")