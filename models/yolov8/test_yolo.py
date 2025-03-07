from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt", "detect")  # load a pretrained model (recommended for training)

# Train the model
print(model)