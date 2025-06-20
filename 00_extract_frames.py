# extract_frames.py
import cv2
import os

bodypart="LAT" 
category="bodyparts"
output_dir = f"{category}/{bodypart}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(f"{bodypart}.mp4")

if not cap.isOpened(): 
    print("ERROR")
    exit()

frame_count = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_filename = os.path.join(output_dir, f'{bodypart}_{frame_count:04d}.jpg')

    cv2.imwrite(frame_filename, frame)
    print(f'Saved {frame_filename}')

    frame_count += 1

cap.release()