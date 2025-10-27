# # import streamlit as st
# # import numpy as np
# # import cv2
# # import os
# # from ultralytics import YOLO
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.utils import load_img, img_to_array
# # import tempfile

# # # ==============================
# # # PATHS & MODEL LOADING
# # # ==============================
# # MODEL_PATH = "traffic_congestion_model.h5"
# # YOLO_MODEL_PATH = "yolov8n.pt"

# # st.set_page_config(page_title="Traffic Congestion Predictor", layout="centered")

# # @st.cache_resource
# # def load_models():
# #     cnn_model = load_model(MODEL_PATH)
# #     yolo_model = YOLO(YOLO_MODEL_PATH)
# #     return cnn_model, yolo_model

# # cnn_model, yolo_model = load_models()

# # class_labels = ["Empty", "Low", "Medium", "High", "Traffic Jam"]

# # # ==============================
# # # UTILITY FUNCTIONS
# # # ==============================
# # def count_vehicles_yolo(img_path):
# #     results = yolo_model(img_path, verbose=False)
# #     vehicle_count = 0
# #     for r in results:
# #         for c in r.boxes.cls:
# #             if int(c) in [2, 3, 5, 7]:  # car, motorbike, bus, truck
# #                 vehicle_count += 1
# #     return vehicle_count

# # def predict_congestion(img_path):
# #     img = load_img(img_path, target_size=(224, 224))
# #     img_array = img_to_array(img) / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)
# #     preds = cnn_model.predict(img_array)
# #     predicted_label = class_labels[np.argmax(preds)]
# #     confidence = np.max(preds)
# #     return predicted_label, confidence

# # # ==============================
# # # STREAMLIT UI
# # # ==============================
# # st.title("🚦 Traffic Congestion Prediction App")
# # st.markdown("""
# # Upload a **road or CCTV image**, and the app will:
# # 1. Detect vehicles using **YOLOv8**
# # 2. Classify congestion level (Low / Medium / High / Jam) using **MobileNetV2**
# # """)

# # uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

# # if uploaded_file:
# #     # Save temporarily
# #     tfile = tempfile.NamedTemporaryFile(delete=False)
# #     tfile.write(uploaded_file.read())
# #     img_path = tfile.name

# #     # Display uploaded image
# #     st.image(img_path, caption="Uploaded Image", use_container_width=True)

# #     # Perform detection and prediction
# #     with st.spinner("Analyzing traffic congestion..."):
# #         vehicle_count = count_vehicles_yolo(img_path)
# #         label, conf = predict_congestion(img_path)

# #         # Draw YOLO boxes
# #         results = yolo_model(img_path, verbose=False)
# #         for r in results:
# #             annotated_img = r.plot()

# #         st.image(annotated_img, caption=f"Detected Vehicles: {vehicle_count}", use_container_width=True)
# #         st.success(f"**Prediction:** {label} ({conf*100:.2f}% confidence)")
# #         st.info(f"🚗 Vehicle Count (YOLOv8): {vehicle_count}")

# # else:
# #     st.warning("Please upload an image to begin.")

# # st.markdown("---")
# # st.caption("Built with ❤️ using Streamlit, YOLOv8, and MobileNetV2")


# # import streamlit as st
# # from ultralytics import YOLO
# # from PIL import Image
# # import tempfile
# # import os

# # # Load YOLO model
# # yolo_model = YOLO("yolov8n.pt")  # or your custom model path

# # st.title("🚗 Traffic Density Detection using YOLOv8")

# # # File upload
# # uploaded_file = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     # Show preview
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# #     # Save temporarily
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
# #         tmp_file.write(uploaded_file.getvalue())
# #         temp_path = tmp_file.name

# #     # Run YOLO detection
# #     st.write("Detecting vehicles... ⏳")
# #     results = yolo_model.predict(temp_path, save=False, verbose=False)

# #     # Show YOLO result
# #     result_img = results[0].plot()  # numpy array
# #     st.image(result_img, caption="Detected Vehicles", use_column_width=True)

# #     # Count vehicles
# #     classes = results[0].boxes.cls.tolist()
# #     vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck (COCO classes)
# #     vehicle_count = sum(1 for c in classes if int(c) in vehicle_classes)

# #     st.success(f"Total Vehicles Detected: {vehicle_count}")

# #     # Clean up temp file
# #     os.remove(temp_path)


# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import tempfile
# import os

# # Load YOLO model
# yolo_model = YOLO("yolov8n.pt")  # small, fast model

# st.set_page_config(page_title="Traffic Density Detection", page_icon="🚗", layout="centered")
# st.title("🚦 Traffic Density Classification using YOLOv8")

# # File uploader
# uploaded_file = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Show preview
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Save temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         temp_path = tmp_file.name

#     # YOLO detection
#     st.write("Detecting vehicles... ⏳")
#     results = yolo_model.predict(temp_path, save=False, verbose=False)

#     # Display result image
#     result_img = results[0].plot()
#     st.image(result_img, caption="Detected Vehicles", use_column_width=True)

#     # Vehicle counting
#     classes = results[0].boxes.cls.tolist()
#     vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck
#     vehicle_count = sum(1 for c in classes if int(c) in vehicle_classes)

#     # Define density levels
#     if vehicle_count <= 5:
#         density = "🟢 Low Traffic"
#         color = "green"
#     elif 6 <= vehicle_count <= 15:
#         density = "🟡 Moderate Traffic"
#         color = "orange"
#     elif 16 <= vehicle_count <= 30:
#         density = "🟠 High Traffic"
#         color = "darkorange"
#     else:
#         density = "🔴 Very High Traffic"
#         color = "red"

#     # Display results
#     st.markdown(f"### 🚗 Total Vehicles Detected: **{vehicle_count}**")
#     st.markdown(f"### 🌆 Traffic Density Level: <span style='color:{color}; font-weight:bold;'>{density}</span>", unsafe_allow_html=True)

#     # Cleanup
#     os.remove(temp_path)
# else:
#     st.info("Please upload a traffic image to analyze 🚘")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from streamlit.components.v1 import html

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Streamlit Page Config
st.set_page_config(page_title="Traffic Congestion Detector", page_icon="🚗", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        color: #36b9cc;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #b0bec5;
        margin-bottom: 30px;
    }
    .result-card {
        background: linear-gradient(145deg, #1c1f26, #20252e);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
        margin-top: 20px;
    }
    .density {
        font-size: 1.6em;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚦 Traffic Congestion Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect vehicles and classify traffic density using YOLOv8</div>", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("📸 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    # YOLO detection
    with st.spinner("🧠 Detecting vehicles... please wait"):
        results = yolo_model.predict(temp_path, save=False, verbose=False)

    # Show YOLO result image
    result_img = results[0].plot()
    st.image(result_img, caption="🎯 YOLOv8 Detection Result", use_container_width=True)

    # Count detected vehicles
    classes = results[0].boxes.cls.tolist()
    vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck
    vehicle_count = sum(1 for c in classes if int(c) in vehicle_classes)

    # Define Density Levels
    if vehicle_count <= 5:
        density = "🟢 Low Traffic"
        color = "#28a745"
        level = "Smooth flow — minimal congestion 🚗💨"
    elif 6 <= vehicle_count <= 15:
        density = "🟡 Moderate Traffic"
        color = "#ffc107"
        level = "Some congestion — steady movement 🚙🚙"
    elif 16 <= vehicle_count <= 30:
        density = "🟠 High Traffic"
        color = "#fd7e14"
        level = "Heavy congestion — possible delays 🚕🚕🚗"
    else:
        density = "🔴 Very High Traffic"
        color = "#dc3545"
        level = "Severe congestion — gridlock likely 🚓🚙🚗🚕"

    # Display results in a stylish card
    st.markdown(f"""
        <div class="result-card">
            <h3 style='color:#61dafb;'>🚗 Total Vehicles Detected: {vehicle_count}</h3>
            <div class="density" style='color:{color};'>{density}</div>
            <p style='color:#b0bec5;'>{level}</p>
        </div>
    """, unsafe_allow_html=True)

    os.remove(temp_path)

else:
    st.info("📤 Please upload a traffic image to analyze.", icon="ℹ️")

# Footer
st.markdown("""
---
<p style="text-align:center; color:grey;">
Built with ❤️ using Streamlit & YOLOv8
</p>
""", unsafe_allow_html=True)
