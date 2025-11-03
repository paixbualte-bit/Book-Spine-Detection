from roboflow import Roboflow
from ultralytics import YOLO

# 1. Download Data from Roboflow
print("Downloading dataset...")
rf = Roboflow(api_key="FfBEBwQuPOekfMmU599Q") # Replace with your key
project = rf.workspace("suaiedu4136").project("bookshlef_detection")
dataset = project.version(5).download("yolov8")
print(f"Dataset downloaded to: {dataset.location}")

# 2. Load a pre-trained model
# yolov8n.pt is the smallest, fastest model
model = YOLO("yolov8n.pt") 

# 3. Train the model
print("Starting training...")
results = model.train(
    data=f"{dataset.location}/data.yaml",  # Path to your dataset's config file
    epochs=75,                            # 75 epochs is a good start
    imgsz=640,                            # Image size (640x640 pixels)
    device=0                              # Use the GPU (device=0)
)

print("Training complete!")
print(results)
