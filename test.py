# Note: If on mac, run 'unalias python' first for interpreter to point to the correct path
from ultralytics import YOLO

print("I'm Running")

# "runs/detect/train/weights/best.pt" MAC
# "runs\\detect\\train\\weights\\best.pt" PC

# Replace model with path to the fine-tuned model trained on the
# 'coco128' dataset
model = YOLO("runs\\detect\\train\\weights\\best.pt")


# Use the fine-tuned model for real-time detection
results = model.predict(source="0", show=True)

print(results)
