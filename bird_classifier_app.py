import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import requests
from io import BytesIO
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_birds_finetuned.keras")

model = load_model()

class_names = [
    'Asian-Green-Bee-Eater', 'Brown-Headed-Barbet', 'Cattle-Egret', 'Common-Kingfisher',
    'Common-Myna', 'Common-Rosefinch', 'Common-Tailorbird', 'Coppersmith-Barbet',
    'Forest-Wagtail', 'Gray-Wagtail', 'Hoopoe', 'House-Crow', 'Indian-Grey-Hornbill',
    'Indian-Peacock', 'Indian-Pitta', 'Indian-Roller', 'Jungle-Babbler',
    'Northern-Lapwing', 'Red-Wattled-Lapwing', 'Ruddy-Shelduck', 'Rufous-Treepie',
    'Sarus-Crane', 'White-Breasted-Kingfisher', 'White-Breasted-Waterhen', 'White-Wagtail'
]

st.title("🦜 Bird Species Classifier")
st.markdown("Upload an image or paste an image URL. The model will identify the bird species.")

st.sidebar.header("Top 3 Predictions")

input_method = st.radio("Choose image input method:", ["Upload from Computer", "Image URL"])

image = None

if input_method == "Upload from Computer":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif input_method == "Image URL":
    url = st.text_input("Paste image URL here:")
    if url:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            if "image" in response.headers.get("Content-Type", ""):
                image = Image.open(BytesIO(response.content))
            else:
                st.error("The URL does not point to an image.")
        except Exception as e:
            st.error(f"Failed to load image from URL: {e}")

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.convert('RGB')
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        predictions = model.predict(img_array)[0]
        top3_indices = predictions.argsort()[-3:][::-1]

        top_class = class_names[top3_indices[0]]
        top_conf = predictions[top3_indices[0]] * 100

        st.markdown(f"### 🐤 Predicted Bird Species: `{top_class}`")
        st.markdown(f"🔢 **Confidence:** `{top_conf:.2f}%`")

        for i in range(1, 3):
            cls = class_names[top3_indices[i]]
            conf = predictions[top3_indices[i]] * 100
            st.sidebar.markdown(f"**{cls}** – `{conf:.2f}%`")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
