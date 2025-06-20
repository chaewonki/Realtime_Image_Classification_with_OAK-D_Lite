# make_mp4.py
import cv2
import depthai as dai

name = "TEST"

pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")

camRgb.preview.link(xoutRgb.input)

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    video_writer = None
    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        cv2.imshow("rgb", frame)

        key = cv2.waitKey(1)
        
        if key == ord('s'):
            print("Start recording")
            video_writer = cv2.VideoWriter(f"{name}.mp4", 
                                           cv2.VideoWriter_fourcc(*'mp4v'), # mp4 코덱
                                           30,                              # 프레임 수
                                           (300,300))                       # preview 디폴트 사이즈
        elif key == ord('e'):
            print("End recording")  
            video_writer.release()
            video_writer = None

        if video_writer is not None:
            video_writer.write(frame)

        if key == ord('q'):
            break

