from ultralytics import YOLO
import supervision as sv
from PIL import Image


# Testing with YOLO Segmentation
model = YOLO("./yolo_seg_train/yolo11n-seg.pt")
image = Image.open("fire_smoke_test_01.jpg")
result = model.predict(image, conf=0.25)[0]
print(result)
# print(result.boxes)

detections = sv.Detections.from_ultralytics(result)
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

annotated_image = image.copy()
annotated_image = mask_annotator.annotate(annotated_image, detections=detections)
annotated_image = label_annotator.annotate(annotated_image, detections=detections)
sv.plot_image(annotated_image, size=(10, 10))