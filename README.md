# Realtime Image Classification with OAK-D Lite
This repository provides a full pipeline for video processing, classification model training using TensorFlow, and deployment on OAK-D Lite using Intel's OpenVINO toolkit.
<br><br>        
## Project Structure
.  
├── README.md  
├── 00_extract_frames.py  
├── 00_make_mp4.py  
├── 01_train_model.py  
├── 02_IR_conversion.sh  
├── 03_make_blob.py  
├── 04_use_blob.py
<br><br>    
## Dependencies
* OpenCV
* DepthAI SDK
* TensorFlow
* OpenVINO Toolkit
<br><br>    
## 1. Recording and Frame Extraction
* 00_make_mp4.py
  * Records video from an OAK-D Lite camera.
  * Press:
    * s: Start recording ({name}.mp4)
    * e: Stop recording
    * q: Quit the app
* 00_extract_frames.py
  * Extract frames from {name}.mp4 and saves them in {category}/{name} as {name}_0001.jpg, {name}_0002.jpg, etc
<br><br>    
## 2. Model Training (TensorFlow)
* Uses MobileNetV2 with custom dense layers.
* Model trained on extracted frames organized by class.
* Early stopping based on validation loss
* Data augmentation
* SaveModel Format
* Visualization using matplotlib
<br><br>    
## 3. Model Optimization
* Converts Tensorflow model into OpenVINO IR format.
<br><br>    
## 4. Conversion to .blob for OAK-D Lite
* Using blobconverter, the IR model is converted into .blob
<br><br>    
## 5. Real-Time Inference on OAK-D Lite
* Captures camera feed
* Displays result on the frame
<br><br>    
