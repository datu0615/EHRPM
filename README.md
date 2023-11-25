# SOFA-FPN

## Train (yolov8)

```bash
from ultralytics import YOLO

# Load a model
model = YOLO('newyolov8s.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='VisDrone.yaml', epochs=100, imgsz=1280)
```


## Train (yolox)

```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o
```
