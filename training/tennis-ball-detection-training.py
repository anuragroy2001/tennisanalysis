from ultralytics import YOLO
# Load a YOLOv8 model pre-trained on COCO dataset
model = YOLO('/home/anurag/TennisAnalysisYOLO/yolov8x.pt')  # You can use a larger version like 'yolov8m.pt'

# Train the model
model.train(data='/home/anurag/TennisAnalysisYOLO/training/tennis-ball-detection-6/data.yaml', epochs=100, imgsz=640)
