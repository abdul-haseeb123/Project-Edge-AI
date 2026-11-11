import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from time import perf_counter_ns
from helper import UltralyticsYOLOClassifier, draw_annotated_bbox
from pathlib import Path
import pandas as pd

ONE_MILLISECOND = 1_000_000 # 1 million nano seconds


base_options = python.BaseOptions(model_asset_path="assets/blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

gender_cls = UltralyticsYOLOClassifier("assets/gender-cls.pt", "assets/gender-labels.txt")
age_cls = UltralyticsYOLOClassifier("assets/age_finetuned-cls.pt", 
                                    "assets/age_finetuned-labels.txt")
expression_cls = UltralyticsYOLOClassifier("assets/expression_finetuned-cls.pt", 
                                           "assets/expression_finetuned-labels.txt")

columns = ["img", "detection", "inference", "total", "gender", "age", "expression", "pred_gender", "pred_age", "pred_expression"]
data = []

# This file is used to generate csv file for all single face photos.

imgs_dir = os.listdir("images")
Path(os.path.join(os.getcwd(), "inferences")).mkdir(parents=True, exist_ok=True)

for img in imgs_dir:
    img_path = os.path.join("images", img)
    img_bgr = cv2.imread(img_path)
    # resized = (1366, 768)

    # img_bgr = cv2.resize(img_bgr, resized, interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    start_detection_time = perf_counter_ns()

    result = detector.detect(mp_img)

    end_detection_time = perf_counter_ns()
    detection_time = (end_detection_time - start_detection_time) // ONE_MILLISECOND
    print("Detection Time:", detection_time, "ms")


    cropped_faces = []

    for detection in result.detections:
        bbox = detection.bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height
        cropped = img_bgr[ y : y + h, x : x + w, :]
        cropped_faces.append(cropped)

    start_inference_time = perf_counter_ns()

    res_gender = [gender_cls.classify(crop) for crop in cropped_faces]
    res_expression = [expression_cls.classify(crop) for crop in cropped_faces]
    res_age = [age_cls.classify(crop) for crop in cropped_faces]

    end_inference_time = perf_counter_ns()
    inference_time = (end_inference_time - start_inference_time) // ONE_MILLISECOND
    print("Inference Time:", inference_time, "ms")

    total_time = detection_time + inference_time
    inference_label = f"Detection {detection_time}ms\nInference {inference_time}ms\nTotal {total_time}ms\n"


    h, w = img_bgr.shape[:2]



    for i in range(len(res_gender)):
        gender = res_gender[i]
        age = res_age[i]
        expression = res_expression[i]
        label = f"{gender} {age} {expression}"
        bbox = result.detections[i].bounding_box


        img_bgr = draw_annotated_bbox(img_bgr, bbox, label, (0, 0, 255), cv2.FONT_HERSHEY_PLAIN, h * 0.002, 3)



    success = cv2.imwrite(f"inferences/{img}", img_bgr)

    if success:
        labels = img.split("_")[:3] # extract all 3 labels from the images
        true_gender = labels[0].lower()
        true_age =  labels[1].lower()
        true_expression = labels[2].lower()

        pred_gender = res_gender[0].split(" ")[0].lower()
        pred_age = res_age[0].split(" ")[0].lower()
        pred_expression = res_expression[0].split(" ")[0].lower()

        data.append([img, detection_time, inference_time, total_time, true_gender, true_age, true_expression, pred_gender, pred_age, pred_expression])

        print("Image successfully saved!")
    else:
        print("Failed to save the image.")



df = pd.DataFrame(data=data, columns=columns)

df.to_csv("inferences.csv")