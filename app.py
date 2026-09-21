import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Detector", page_icon="🔍", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_URL = "https://ultralytics.com/images/bus.jpg"


# ---------- Model (loaded once, reused across reruns) ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model(name: str) -> YOLO:
    local = APP_DIR / name
    # Use the local .pt if it exists; otherwise ultralytics auto-downloads it
    return YOLO(str(local) if local.exists() else name)


# ---------- Image loading ----------
def load_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ---------- Sidebar controls ----------
st.sidebar.header("Settings")

model_name = st.sidebar.selectbox(
    "Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    help="n = fastest, m = most accurate. Non-local models download on first use.",
)
model = load_model(model_name)

conf = st.sidebar.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 1.0, 0.45, 0.05)

all_classes = list(model.names.values())
selected = st.sidebar.multiselect("Only detect these classes (empty = all)", all_classes)
class_ids = [k for k, v in model.names.items() if v in selected] or None

source = st.sidebar.radio("Image source", ["Upload", "URL", "Webcam snapshot"])

# ---------- Main page ----------
st.title("🔍 YOLOv8 Object Detector")

image = None
try:
    if source == "Upload":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if file:
            image = Image.open(file).convert("RGB")
    elif source == "URL":
        url = st.text_input("Image URL", DEFAULT_URL)
        if url:
            image = load_from_url(url)
    else:
        snap = st.camera_input("Take a photo")
        if snap:
            image = Image.open(snap).convert("RGB")
except Exception as e:
    st.error(f"Couldn't load the image: {e}")

if image is None:
    st.info("Pick an image source to get started.")
    st.stop()

# ---------- Inference ----------
with st.spinner("Detecting..."):
    # Pass the PIL image directly (treated as RGB). A raw numpy array would be assumed BGR.
    results = model.predict(image, conf=conf, iou=iou, classes=class_ids, verbose=False)
r = results[0]

# r.plot() returns a BGR numpy array -> flip to RGB for display
annotated = Image.fromarray(r.plot()[:, :, ::-1])

col1, col2 = st.columns(2)
col1.subheader("Original")
col1.image(image, use_container_width=True)
col2.subheader("Detections")
col2.image(annotated, use_container_width=True)

# ---------- Results table + stats ----------
rows = []
for cls_id, score, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
    rows.append({
        "class": model.names[int(cls_id)],
        "confidence": round(score, 3),
        "x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3]),
    })
df = pd.DataFrame(rows)

m1, m2, m3 = st.columns(3)
m1.metric("Objects found", len(df))
m2.metric("Unique classes", df["class"].nunique() if not df.empty else 0)
m3.metric("Inference time", f"{r.speed['inference']:.0f} ms")

if df.empty:
    st.warning("Nothing detected. Try lowering the confidence threshold.")
else:
    left, right = st.columns([2, 1])
    left.dataframe(df, use_container_width=True, hide_index=True)
    right.bar_chart(df["class"].value_counts())

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button("⬇️ Download annotated image", buf.getvalue(),
                       file_name="detections.png", mime="image/png")