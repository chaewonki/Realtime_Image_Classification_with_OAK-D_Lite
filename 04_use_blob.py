import depthai as dai
import cv2
import numpy as np

blob_file_path = './outputs/hand_model.blob'
class_names = ['LAT', 'PA']

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip.initialConfig.setResize(224, 224)
manip.initialConfig.setResizeThumbnail(224, 224, 3)
manip.setKeepAspectRatio(True)

cam.preview.link(manip.inputImage)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blob_file_path)
nn.input.setBlocking(False)

manip.out.link(nn.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn_output")
nn.out.link(xout_nn.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb_output")
cam.preview.link(xout_rgb.input)

with dai.Device(pipeline) as device:
    nn_queue = device.getOutputQueue(name="nn_output", maxSize=1, blocking=False)
    
    video_queue = device.getOutputQueue(name="rgb_output", maxSize=1, blocking=False)

    while True:
        frame = video_queue.get().getCvFrame()
        
        nn_data = nn_queue.get()
        detections = nn_data.getFirstLayerFp16()

        if detections:
            predicted_class = np.argmax(detections)
            confidence = detections[predicted_class]
            label = f"Class: {class_names[predicted_class]}, Conf: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("OAK-D Lite Camera Feed", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()