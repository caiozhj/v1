from ultralytics import YOLO

model = YOLO("yolo11x-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
