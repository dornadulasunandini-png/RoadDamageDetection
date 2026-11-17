
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🚧 Road Damage Detection System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered road surface damage identification using YOLO11</div>',
    unsafe_allow_html=True
)

st.divider()

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Sidebar
st.sidebar.title("⚙️ Project Information")
st.sidebar.write("**Model:** YOLO11n")
st.sidebar.write("**Task:** Object Detection")
st.sidebar.write("**Framework:** PyTorch + Ultralytics")
st.sidebar.write("**Platform:** Streamlit Cloud")
st.sidebar.divider()
st.sidebar.info(
    "Upload a road image to identify possible road damage regions."
)

# Upload section
st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.success("Image uploaded successfully!")

    if st.button("🔍 Detect Road Damage", use_container_width=True):

        with st.spinner("AI model is analyzing the road image..."):

            temp_path = None

            try:
                model = load_model()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp_file:
                    image.save(temp_file.name)
                    temp_path = temp_file.name

                results = model.predict(
                    source=temp_path,
                    conf=0.25,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                annotated_image = result.plot()
                annotated_image = Image.fromarray(
                    annotated_image[:, :, ::-1]
                )

                boxes = result.boxes
                total = len(boxes)

                # Results
                st.divider()
                st.subheader("📊 Detection Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Detections", total)

                with col2:
                    damage_classes = (
                        len(set(int(c) for c in boxes.cls.tolist()))
                        if total else 0
                    )
                    st.metric("Damage Classes", damage_classes)

                with col3:
                    average_confidence = (
                        float(boxes.conf.mean()) * 100
                        if total else 0
                    )
                    st.metric(
                        "Average Confidence",
                        f"{average_confidence:.1f}%"
                    )

                st.divider()

                left, right = st.columns(2)

                with left:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)

                with right:
                    st.subheader("Detected Road Damage")
                    st.image(
                        annotated_image,
                        use_container_width=True
                    )

                st.subheader("🔎 Detection Details")

                if total > 0:
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0]) * 100

                        st.write(
                            f"**Detection {i + 1}:** "
                            f"{class_name} — "
                            f"{confidence:.2f}% confidence"
                        )
                else:
                    st.warning(
                        "No road damage was detected in this image. "
                        "This does not guarantee that the road is damage-free."
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

st.divider()

st.markdown(
    "<center>🚧 Road Damage Detection | AI & Machine Learning Project</center>",
    unsafe_allow_html=True
)
