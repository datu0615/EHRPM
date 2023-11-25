## Train (yolov8)

```bash
from ultralytics import YOLO

# Load a model
model = YOLO('newyolov8s.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='VisDrone.yaml', epochs=100, imgsz=1280)
```
