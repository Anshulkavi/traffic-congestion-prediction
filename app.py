# # # import streamlit as st
# # # from PIL import Image
# # # import numpy as np
# # # import tensorflow as tf

# # # # Load trained model
# # # model = tf.keras.models.load_model("traffic_cnn_model.h5")

# # # # Labels (same as training)
# # # CLASS_NAMES = ["Low", "Medium", "High"]

# # # def preprocess_image(image, img_size=224):
# # #     image = image.convert("RGB").resize((img_size, img_size))
# # #     img_array = np.array(image) / 255.0
# # #     img_array = np.expand_dims(img_array, axis=0)
# # #     return img_array

# # # st.set_page_config(page_title="Smart Traffic Flow Prediction", layout="centered")

# # # st.title("🚦 Smart Traffic Flow Prediction")
# # # st.markdown("Upload a **CCTV or road image** to predict congestion level.")

# # # uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# # # if uploaded_file is not None:
# # #     image = Image.open(uploaded_file)
# # #     st.image(image, caption="Uploaded Image", use_column_width=True)
    
# # #     if st.button("Predict Traffic Level"):
# # #         with st.spinner("Analyzing traffic..."):
# # #             img_array = preprocess_image(image)
# # #             preds = model.predict(img_array)
# # #             pred_class = np.argmax(preds, axis=1)[0]
# # #             confidence = float(np.max(preds))
# # #             label = CLASS_NAMES[pred_class]

# # #         st.success(f"**Predicted Congestion:** {label}")
# # #         st.write(f"Confidence: {confidence:.2f}")


# # import streamlit as st
# # import numpy as np
# # from PIL import Image
# # import tensorflow as tf
# # import time

# # # =========================
# # # 🔹 Load Trained Model
# # # =========================
# # @st.cache_resource
# # def load_model():
# #     return tf.keras.models.load_model("traffic_cnn_model.h5")

# # model = load_model()

# # # =========================
# # # 🔹 Define Labels
# # # =========================
# # CLASS_NAMES = ["Low", "Medium", "High"]

# # # =========================
# # # 🔹 Preprocessing Function
# # # =========================
# # def preprocess_image(image, img_size=224):
# #     image = image.convert("RGB").resize((img_size, img_size))
# #     img_array = np.array(image) / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)
# #     return img_array

# # # =========================
# # # 🔹 Streamlit UI
# # # =========================
# # st.set_page_config(page_title="Smart Traffic Flow Prediction", layout="centered")

# # st.title("🚦 Smart Traffic Flow Prediction")
# # st.markdown(
# #     """
# #     Upload a **CCTV or road image** to predict congestion level.  
# #     The AI model (MobileNetV2) analyzes the road density to classify:
# #     - 🟢 **Low Traffic**
# #     - 🟡 **Medium Traffic**
# #     - 🔴 **High Traffic**
# #     """
# # )

# # uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded Image", use_container_width=True)
# #     st.write("")

# #     if st.button("🚀 Predict Traffic Level"):
# #         with st.spinner("Analyzing traffic..."):
# #             time.sleep(1.5)  # short delay for UX
# #             img_array = preprocess_image(image)
# #             preds = model.predict(img_array)
# #             pred_class = np.argmax(preds, axis=1)[0]
# #             confidence = float(np.max(preds))
# #             label = CLASS_NAMES[pred_class]

# #         st.success(f"**Predicted Congestion Level:** {label}")
# #         st.progress(confidence)
# #         st.write(f"**Confidence:** {confidence*100:.2f}%")

# #         # Visualization bar
# #         st.subheader("Prediction Confidence Breakdown:")
# #         st.bar_chart(
# #             {
# #                 "Low": preds[0][0],
# #                 "Medium": preds[0][1],
# #                 "High": preds[0][2],
# #             }
# #         )

# # st.markdown("---")
# # st.caption("Developed with ❤️ using TensorFlow + Streamlit")


# import streamlit as st
# from PIL import Image
# import tensorflow as tf
# import numpy as np
# import os

# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("traffic_cnn_model.h5")

# model = load_model()

# CLASS_NAMES = ["Low", "Medium", "High"]
# IMG_SIZE = 64  # match small CNN

# st.title("Traffic Congestion Prediction 🚦")
# uploaded_file = st.file_uploader("Upload a CCTV/Road Image", type=["jpg","jpeg","png"])

# def predict_congestion(image, model, img_size=IMG_SIZE):
#     image = image.convert("RGB").resize((img_size, img_size))
#     img_array = np.array(image)/255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     preds = model.predict(img_array)
#     pred_class = np.argmax(preds, axis=1)[0]
#     confidence = float(np.max(preds))
#     return CLASS_NAMES[pred_class], confidence

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_container_width=True)
    
#     cls, conf = predict_congestion(image, model)
#     st.write(f"**Predicted Congestion:** {cls} ({conf*100:.2f}%)")


import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
CLASS_NAMES = ["Low", "Medium", "High"]

st.title("Traffic Congestion Prediction 🚦")

model = tf.keras.models.load_model("traffic_cnn_mobilenetv2.h5")

def predict_congestion(image, model, target_size=(IMG_SIZE, IMG_SIZE)):
    image_resized = image.convert("RGB").resize(target_size)
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    
    return CLASS_NAMES[pred_class], confidence

uploaded_file = st.file_uploader("Upload a CCTV/Road Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    cls, conf = predict_congestion(image, model)
    st.write(f"**Predicted Congestion:** {cls} ({conf*100:.2f}%)")
