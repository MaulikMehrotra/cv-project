from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Provide the path to an image. 
# You can use a local file (e.g., "my_photo.jpg") or a direct web URL!
image_source = "https://ultralytics.com/images/bus.jpg"

# 3. Run the model on the image
results = model(image_source)

# 4. Display the result in a pop-up window
# The .show() method is a built-in Ultralytics feature that handles the display for you
results[0].show()

print("Close the image window to end the program.")