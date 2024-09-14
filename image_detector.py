import os
import time
import torch
import cv_bridge
import cv2
import PIL
import numpy as np

bridge = cv_bridge.CvBridge()

path_hubconfig = os.path.expanduser("~/yolov5")
model = torch.hub.load(
    path_hubconfig, "custom", path="./yolov5m.pt", source="local"
)
model.conf = 0.25
model.iou = 0.45

img = PIL.Image.open("./fridge.jpg")
img = np.array(img.convert("RGB"))

results = model(img)
preds = results.pandas().xyxy[0]
print(preds)

for _, row in preds.iterrows():
    min_x = int(row["xmin"])
    min_y = int(row["ymin"])
    max_x = int(row["xmax"])
    max_y = int(row["ymax"])

    label = row["class"]
    name = row["name"]
    probability = row["confidence"]
    print(name, probability)
    cv2.rectangle(
        img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 4
    )

cv2.imwrite("./output.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
