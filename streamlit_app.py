import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from inference_sdk import InferenceHTTPClient
import io
import base64

# Initialize Roboflow Inference Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="rjlwQNV7ngvYcztnjlyB"
)

# Streamlit UI
st.set_page_config(page_title="Crowd Detection App", layout="wide")
st.title("🚶 Crowd Detection Application")
st.sidebar.header("Upload Input")

# Input options
input_type = st.sidebar.radio("Select Input Type:", ["Image", "Video", "Live Camera"], key="input_type_radio")

# Create a session state variable for debugging
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Add debug toggle in sidebar (outside of any loop)
st.sidebar.checkbox("Show API Response", key="debug_checkbox", value=st.session_state.debug_mode, 
                   on_change=lambda: setattr(st.session_state, 'debug_mode', not st.session_state.debug_mode))

def process_image(image, frame_id=0):
    image_pil = Image.fromarray(image)
    
    # Call the Roboflow workflow
    result = client.run_workflow(
        workspace_name="dnyaneshwar",
        workflow_id="detect-count-and-visualize-2",
        images={
            "image": image_pil
        },
        use_cache=True 
    )
    
    # Use the session state for debugging instead of creating a new widget
    if st.session_state.debug_mode and frame_id == 0:  # Only show for the first frame
        st.sidebar.json(result)
    
    # Extract predictions based on the format provided
    predictions = []
    people_count = 0
    
    if isinstance(result, list) and len(result) > 0:
        if "predictions" in result[0] and "predictions" in result[0]["predictions"]:
            predictions = result[0]["predictions"]["predictions"]
            people_count = result[0].get("count_objects", len(predictions))
    
    # Draw bounding boxes
    for prediction in predictions:
        if "x" in prediction and "y" in prediction and "width" in prediction and "height" in prediction:
            x, y, w, h = map(float, [prediction["x"], prediction["y"], prediction["width"], prediction["height"]])
            # Convert to integers for drawing
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add class label if available
            if "class" in prediction:
                label = prediction["class"].split(" - ")[0]  # Shorten the class name
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Add confidence score if available
            if "confidence" in prediction:
                conf = prediction["confidence"]
                cv2.putText(image, f"{conf:.2f}", (x2-30, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Create a semi-transparent overlay for the count display
    overlay = image.copy()
    # Draw a black semi-transparent rectangle at the top
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
    # Apply the overlay with transparency
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Show count of detected people
    cv2.putText(image, f"People Count: {people_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Return both the image and the count
    return image, people_count

if input_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="image_uploader")
    if uploaded_image:
        image = Image.open(uploaded_image)
        image = np.array(image)
        
        # Create a counter display
        count_display = st.empty()
        
        # Process the image
        processed_image, people_count = process_image(image.copy())
        
        # Display the count in a prominent way
        count_display.markdown(f"""
        <div style="background-color:#1E88E5; padding:10px; border-radius:5px; margin-bottom:10px;">
            <h2 style="color:white; text-align:center;">👥 People Detected: {people_count}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(processed_image, caption="Detected Crowds", use_column_width=True)

elif input_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        # Video processing controls
        st.sidebar.subheader("Video Processing Settings")
        frame_skip = st.sidebar.slider("Process every N frames", 1, 10, 3, key="frame_skip_slider")
        
        # Create a counter display
        count_display = st.empty()
        video_display = st.empty()
        
        video = cv2.VideoCapture(tfile.name)
        
        frame_count = 0
        stop_button = st.button("Stop Processing", key="stop_video_button")
        
        while video.isOpened() and not stop_button:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, people_count = process_image(frame.copy(), frame_count)
            
            # Update the counter display
            count_display.markdown(f"""
            <div style="background-color:#1E88E5; padding:10px; border-radius:5px; margin-bottom:10px;">
                <h2 style="color:white; text-align:center;">👥 People Detected: {people_count}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the processed frame
            video_display.image(processed_frame, caption="Processed Video", use_column_width=True)
            
        video.release()

elif input_type == "Live Camera":
    st.sidebar.subheader("Camera Settings")
    run_camera = st.sidebar.checkbox("Start Camera", False, key="start_camera_checkbox")
    
    if run_camera:
        # Create a counter display
        count_display = st.empty()
        camera_display = st.empty()
        
        cap = cv2.VideoCapture(0)
        stop_button = st.button("Stop Camera", key="stop_camera_button")
        
        frame_count = 0
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, people_count = process_image(frame.copy(), frame_count)
            
            # Update the counter display
            count_display.markdown(f"""
            <div style="background-color:#1E88E5; padding:10px; border-radius:5px; margin-bottom:10px;">
                <h2 style="color:white; text-align:center;">👥 People Detected: {people_count}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the processed frame
            camera_display.image(processed_frame, caption="Live Feed", use_column_width=True)
            
        cap.release()

# Additional information
st.sidebar.info("This app detects crowded areas in images, videos, and live feeds using Roboflow Inference API.")

# Add a metrics section
st.sidebar.markdown("---")
st.sidebar.subheader("Understanding the Metrics")
st.sidebar.markdown("""
- **People Count**: The total number of people detected in the frame
- **Confidence**: The model's confidence level in each detection (0-1)
""")
