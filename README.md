YOLOv8 Object Detector

A simple, beginner-friendly computer vision project that uses YOLOv8, one of the most popular real-time object detection models, to identify and classify objects in images and highlight them with labeled bounding boxes.

The project is wrapped in a Streamlit web app so you can interact with the model directly: try different image sources and tune the model's parameters to see how they change the results.

Show Image

Features
Multiple input sources: upload an image, paste an image URL, or take a webcam snapshot
Adjustable model parameters: confidence threshold, IoU (NMS) threshold, and model size (yolov8n / yolov8s / yolov8m)
Class filtering: detect only the objects you care about (e.g. just person and car)
Visual results: original and annotated images side by side
Detection details: table of classes, confidence scores, and box coordinates, plus a class-count chart and inference time
Download: save the annotated image as a PNG
Project structure
.
├── app.py          # Streamlit web app
├── detector.py     # Minimal script: run YOLOv8 on one image
├── yolov8n.pt      # Pre-trained YOLOv8 nano weights
├── requirements.txt
└── README.md
Getting started
bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

To try the bare-bones version without the web UI: python detector.py

How it works
The image is loaded from the chosen source.
YOLOv8 runs a single forward pass and predicts bounding boxes, class labels, and confidence scores.
Predictions below the confidence threshold are dropped, and overlapping boxes are merged using non-max suppression (controlled by the IoU threshold).
The remaining detections are drawn on the image and shown in the app.

The pre-trained model recognizes the 80 classes of the COCO dataset (person, car, dog, bus, chair, and more).

Tech stack

Python · Ultralytics YOLOv8 · Streamlit · Pillow · pandas

Possible improvements
Video file and live webcam detection
Custom-trained model on your own dataset
Object tracking across video frames
