from ultralytics import YOLO

# base model 
base_model = "models/yolov8l.pt"

# dataset YAML
data_yaml = "data.yaml"       

# Training parameters
epochs   = 100
img_size = 640
batch    = 16
project  = "runs/detect"
name     = "train"

# Load the base model
model = YOLO(base_model)

model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch,
    project=project,
    name=name
)
