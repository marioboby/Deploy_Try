import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Egyptian Food Detector", page_icon="🍽️")

st.title("🍽️ Egyptian Food Detection")
st.write("Upload an image to detect: **Bechamel, Molokhya, Koshary, etc.**")

# 2. Load Model (Cache it so it doesn't reload on every action)
@st.cache_resource
def load_model():
    # Update this path if you move the file!
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Check if the path is correct: {e}")

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # 4. Run Detection when button is clicked
    if st.button("Detect Food"):
        with st.spinner("Analyzing..."):
            # Run inference
            results = model(image)
            
            # Plot results (returns a numpy array in BGR format)
            res_plotted = results[0].plot()
            
            # Display results in the second column
            with col2:
                # 'channels="BGR"' corrects the colors for display
                st.image(res_plotted, caption="Detected Food", channels="BGR", use_container_width=True)
            
            # Optional: Show confidence scores
            st.success("Detection Complete!")
            boxes = results[0].boxes
            if boxes:
                st.write("### Detected Items:")
                for box in boxes:
                    # Get class ID and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    st.write(f"- **{name}**: {conf:.2%} confidence")
            else:
                st.warning("No Egyptian food detected in this image.")