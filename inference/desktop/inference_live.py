import cv2
import mediapipe as mp
import os
import threading
from helper import UltralyticsYOLOClassifier, draw_annotated_bbox, draw_inferences
from time import perf_counter_ns

lock = threading.Lock()

FONT_FACE = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.75
THICKNESS = 1
BBOX = (89, 94, 17) # BGR

PANEL_W = 150
PANEL_H = 80
MARGIN = 10

gender_cls = UltralyticsYOLOClassifier("assets/gender-cls.pt", "assets/gender-labels.txt", stream=True)
age_cls = UltralyticsYOLOClassifier("assets/age_finetuned-cls.pt", 
                                    "assets/age_finetuned-labels.txt", stream=True)
expression_cls = UltralyticsYOLOClassifier("assets/expression_finetuned-cls.pt", 
                                           "assets/expression_finetuned-labels.txt", stream=True)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

result_from_callback = []
result_classifiers = {}
inference_label = None

# Create a face detector instance with the live stream mode:
def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    # converting from rgb to bgr for yolo
    np_img = output_image.numpy_view()[..., ::-1]
    global result_from_callback
    global result_classifiers
    global inference_label

    with lock:
        end_detection_time = perf_counter_ns()
        detection_time = (end_detection_time - timestamp_ms) // 1_000_000

        cropped_faces = []
        for detection in result.detections:
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            w, h = bbox.width, bbox.height
            cropped = np_img[ y : y + h, x : x + w, :]
            cropped_faces.append(cropped)
    
        res_gender = [gender_cls.classify(crop) for crop in cropped_faces]
        res_expression = [expression_cls.classify(crop) for crop in cropped_faces]
        res_age = [age_cls.classify(crop) for crop in cropped_faces]

        end_time = perf_counter_ns()
        inference_time = (end_time - end_detection_time) // 1_000_000
        total_time = detection_time + inference_time

        inference_label = f"Detection {detection_time}ms\nInference {inference_time}ms\nTotal {total_time}ms\n"


        result_classifiers["age"] = res_age
        result_classifiers["gender"] = res_gender
        result_classifiers["expression"] = res_expression
        result_from_callback = result

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Not found webcam")
    exit()

# Get camera FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Sometimes webcams return 0 or an unreliable FPS
if fps <= 0:
    fps = 30.0

frame_index = 0
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=os.path.join("assets", "blaze_face_short_range.tflite")),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback)

with FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("no frame found ")
            break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame[..., ::-1])
        frame_timestamp_ms = perf_counter_ns()
        detector.detect_async(mp_image, frame_timestamp_ms)


        with lock:
            result = result_from_callback
            result_cls = result_classifiers
            infer_label = inference_label

        if result and result_cls:

            if infer_label:
                frame = draw_inferences(frame, infer_label, BBOX, FONT_FACE, 1, THICKNESS, PANEL_W, PANEL_H, MARGIN)

            for i in range(len(result.detections)):

                gender = result_cls["gender"][i]
                age = result_cls["age"][i]
                expression = result_cls["expression"][i]
                label = f"{gender} {age} {expression}"
                bbox = result.detections[i].bounding_box

                frame = draw_annotated_bbox(frame, bbox, label, BBOX, FONT_FACE, FONT_SCALE, THICKNESS)

        frame_index += 1
        cv2.imshow("webfeed", frame)

        # 4. Wait for 1 millisecond and check if the 'q' key is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()