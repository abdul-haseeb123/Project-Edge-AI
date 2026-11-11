# EdgeAI Face Attribute Recognition

An end-to-end computer-vision project that detects faces and predicts three attributes for each face:

- Gender: `Female` or `Male`
- Age group: `Adult` or `Elderly`
- Expression: `Happy` or `Sad`

The project supports both desktop inference and on-device Android inference. Face detection is performed with MediaPipe BlazeFace, followed by three Ultralytics YOLO classification models.

## Pipeline

```text
Camera image or photo
        |
        v
MediaPipe face detection
        |
        v
Face crop for each detection
        |
        +--> Gender classification
        +--> Age classification
        +--> Expression classification
        |
        v
Annotated image or live preview
```

## Project structure

```text
training/
  age-gender-training/       Age and gender training notebooks/results
  expression-training/       Expression training notebooks/results

inference/
  desktop/
    assets/                  PyTorch classifiers and face detector
    images/                  Sample input images
    inferences/              Annotated image output
    inference_img.py         Still-image inference and CSV evaluation
    inference_live.py        Real-time webcam inference
    helper.py                Shared classifier and drawing utilities
    inferences.csv           Image predictions and timing results
    inferences_edge.csv      Timing measurements

  edge/app/
    app/src/main/             Android application source
    app/src/main/assets/     TensorFlow Lite models and label files
```

## Models

The desktop models are stored in [`inference/desktop/assets`](./inference/desktop/assets):

- `blaze_face_short_range.tflite` — face detector
- `gender-cls.pt` — gender classifier
- `age_finetuned-cls.pt` — age classifier
- `expression_finetuned-cls.pt` — expression classifier

The Android app uses TensorFlow Lite versions of the classifiers in [`inference/edge/app/app/src/main/assets`](./inference/edge/app/app/src/main/assets).

The training notebooks use FairFace-derived data and Ultralytics classification models. Training configurations and metrics are preserved in the `*-runs` directories.

## Desktop inference

### Requirements

Install Python 3 and the packages used by the desktop scripts:

```bash
pip install ultralytics opencv-python mediapipe pandas
```

Run commands from the desktop inference directory:

```bash
cd inference/desktop
```

### Process sample images

Place images in `inference/desktop/images/` and use the filename format:

```text
<gender>_<age>_<expression>_<number>.JPG
```

Example:

```text
female_adult_happy_13.JPG
```

Run:

```bash
python inference_img.py
```

The script detects faces, runs all three classifiers, saves annotated images to `inferences/`, and writes predictions and timing data to `inferences.csv`.

### Run webcam inference

```bash
python inference_live.py
```

Press `q` to exit the webcam window.

## Android edge application

The Android application is located at [`inference/edge/app`](./inference/edge/app). It performs inference locally on the device using CameraX, MediaPipe, LiteRT, and Jetpack Compose.

To build the debug APK:

```bash
cd inference/edge/app
gradlew.bat assembleDebug
```

The resulting APK is generated under:

```text
app/build/outputs/apk/debug/
```

The app supports:

- Live camera face detection and classification
- Front and back camera switching
- Selecting an image from the gallery
- Capturing and saving annotated photos
- Displaying detection and classification timing

Camera permission is required at runtime.

## Important notes

- The models predict broad categories and should not be treated as definitive demographic or emotional assessments.
- Accuracy and latency depend on lighting, pose, image quality, hardware, and face size.
- The sample CSV evaluation is intended for demonstration and smoke testing, not as a statistically representative benchmark.
- Desktop inference uses `.pt` models, while Android inference uses converted `.tflite` models. Their preprocessing must remain compatible for comparable results.