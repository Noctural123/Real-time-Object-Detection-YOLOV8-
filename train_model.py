from ultralytics import YOLO

# Load the pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Fine-tune the model on your custom dataset
model.train(data="coco128.yaml")

# Use the fine-tuned model for real-time detection
results = model.predict(source=0, show=True)

print(results)
