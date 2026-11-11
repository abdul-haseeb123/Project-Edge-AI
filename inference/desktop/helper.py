from ultralytics import YOLO
import cv2

class UltralyticsYOLOClassifier:
    def __init__(self, model_asset: str, labels_asset: str, stream=False):
        self.labels = []
        self.model = YOLO(model_asset)
        self.stream = stream

        self.load_labels(labels_asset)

    def load_labels(self, labels_asset):
        with open(labels_asset, 'r') as f:
            lines = f.readlines()
            self.labels = [line.split("\n")[0] for line in lines]

    def classify(self, face_crop ) -> str:
        res = self.model(face_crop, stream=self.stream)
        # res will only output 1 prediction as only 1 face would be passed
        pred = None

        if self.stream:
            for p in res:
                pred = p
                break
        else:
            pred = res[0]
        label = self.labels[pred.probs.top1]
        return f"{label} {pred.probs.top1conf:.2f}"


def draw_inferences(image, label, color, font_face, font_scale, thickness, panel_w, panel_h, margin):
    h, w = image.shape[:2]

    x1 = max(0, w - panel_w - margin)
    y1 = margin
    x2 = min(w - 1, w - margin)
    y2 = min(h - 1, y1 + panel_h)

    # Draw panel background
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    # Panel text
    text_x = x1 + 10
    text_y = y1 + 25

    image = cv2.putText(image, label,
                (text_x, text_y),
                font_face, font_scale, (255, 255, 255), thickness)
    return image

def draw_annotated_bbox(image, bbox, label, color, font_face, font_scale, thickness):
    pt1 = bbox.origin_x, bbox.origin_y
    pt2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

    cv2.rectangle(image, pt1, pt2, color, thickness=thickness)
    (label_bbox_width, label_bbox_height), baseline = cv2.getTextSize(label, font_face, font_scale, thickness)

    pt1_text_box = (pt1[0], pt1[1] - label_bbox_height)
    pt2_text_box = (pt1[0] + label_bbox_width, pt1[1])

    cv2.rectangle(image, pt1_text_box, pt2_text_box, color, -1, cv2.LINE_AA)

    center_text_box = ((pt1_text_box[0] + pt2_text_box[0]) // 2), ((pt1_text_box[1] + pt2_text_box[1]) // 2)
    text_x = center_text_box[0] - (label_bbox_width // 2)
    text_y = (center_text_box[1] + (label_bbox_height // 2)) - baseline

    cv2.putText(image, label,  (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness=thickness)
    return image