# Realtime Image Classification with OAK-D Lite
Real-time **hand / foot image classification** on an OAK-D Lite camera.

This repository covers the full pipeline: recording training videos with the OAK-D Lite, extracting frames, training a MobileNetV2-based classifier with TensorFlow, converting it with Intel's OpenVINO toolkit, and running real-time on-device inference.
<br><br>

## Classes
| Class | Description |
|-------|-------------|
| Hand  | Images of a hand |
| Foot  | Images of a foot |

Labels are taken from the sub-folder names under `bodyparts/` (sorted alphabetically by TensorFlow). The `class_names` list in `04_use_blob.py` must match that order.
<br><br>

## Project Structure
```
.
├── README.md
├── 00_make_mp4.py         # Record video from the OAK-D Lite
├── 00_extract_frames.py   # Split a video into JPG frames
├── 01_train_model.py      # Train the hand/foot classifier (TensorFlow)
├── 02_IR_conversion.sh    # SavedModel -> OpenVINO IR (.xml/.bin)
├── 03_make_blob.py        # OpenVINO IR -> .blob for the OAK-D Lite
└── 04_use_blob.py         # Real-time inference on the OAK-D Lite
```

Generated (git-ignored) directories:
```
bodyparts/    # Training images, one sub-folder per class
hand_model/   # TensorFlow SavedModel
outputs/      # OpenVINO IR files and the final .blob
```
<br><br>

## Dependencies
* Python 3
* OpenCV (`opencv-python`)
* DepthAI SDK (`depthai`)
* TensorFlow
* Matplotlib
* NumPy
* OpenVINO Toolkit (Model Optimizer `mo`)
* blobconverter
<br><br>

## 1. Recording and Frame Extraction
* `00_make_mp4.py`
  * Records the OAK-D Lite RGB preview (300x300, 30 fps) to `{name}.mp4`.
  * Set `name` in the script (e.g. the class name) before running.
  * Keys:
    * `s`: Start recording
    * `e`: Stop recording
    * `q`: Quit
* `00_extract_frames.py`
  * Extracts every frame from `{bodypart}.mp4` and saves them to `bodyparts/{bodypart}/` as `{bodypart}_0001.jpg`, `{bodypart}_0002.jpg`, ...
  * Set `bodypart` in the script and run it once per class (hand, foot).
<br><br>

## 2. Model Training (TensorFlow) — `01_train_model.py`
* Loads images from `bodyparts/` at 224x224 with an 80/20 train/validation split.
* Pixel values rescaled to [0, 1].
* Backbone: ImageNet-pretrained MobileNetV2 (frozen).
* Head: GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(2, softmax).
* Adam (lr = 1e-5), sparse categorical cross-entropy, up to 20 epochs.
* Early stopping on validation loss (patience 3, best weights restored).
* Saves the model in SavedModel format to `hand_model/`.
* Plots training/validation accuracy and loss with matplotlib.
<br><br>

## 3. Model Optimization — `02_IR_conversion.sh`
* Converts the TensorFlow SavedModel into OpenVINO IR (`saved_model.xml` / `saved_model.bin`).
* FP16 compression, input shape `[1,224,224,3]`, NHWC → NCHW layout, `--scale 255` (so the camera can feed raw 0–255 pixels).
* Move the generated IR files into `outputs/` before the next step (or change `--output_dir`).
<br><br>

## 4. Conversion to .blob for OAK-D Lite — `03_make_blob.py`
* Uses blobconverter (FP16, 4 shaves, OpenVINO 2022.1) to compile `outputs/saved_model.xml/.bin` into `outputs/hand_model.blob`.
<br><br>

## 5. Real-Time Inference on OAK-D Lite — `04_use_blob.py`
* Captures the color camera feed and resizes it to 224x224 (BGR planar) on-device with `ImageManip`.
* Runs the `.blob` on the OAK-D Lite's neural network node.
* Overlays the predicted class (hand / foot) and confidence on the live frame.
* Press `q` to quit.
<br><br>

## Quick Start
```bash
python 00_make_mp4.py          # record a hand video, then a foot video
python 00_extract_frames.py    # run once per class
python 01_train_model.py
bash 02_IR_conversion.sh       # then move the IR files into outputs/
python 03_make_blob.py
python 04_use_blob.py
```
