# camera_video.py (FULL FIXED VERSION)

import cv2 as cv
import time
import geocoder
import os
import sys

# ----------------------------
#  Create result folder safely
# ----------------------------
os.makedirs("pothole_coordinates", exist_ok=True)

# ----------------------------
#  Load class names
# ----------------------------
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# ----------------------------
#  Load YOLOv4-Tiny model
# ----------------------------
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights',
                      'project_files/yolov4_tiny.cfg')

# Try CUDA → if error, fallback to CPU
use_cuda = False
try:
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    use_cuda = True
    print("✓ Using CUDA backend for DNN.")
except:
    print("⚠ CUDA unavailable → Switching to CPU.")
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# ----------------------------
#  Load input video
# ----------------------------
video_path = r"test.mp4"   # change to full path if needed

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ ERROR: Cannot open video file → {video_path}")
    sys.exit(1)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("❌ ERROR: Video width/height = 0. Wrong file or corrupted video.")
    sys.exit(1)

print("✓ Video loaded: {width} x {height}  |  CUDA: {use_cuda}")

# ----------------------------
#  Output VideoWriter
# ----------------------------
output_path = r"result.avi"
fourcc = cv.VideoWriter_fourcc(*'MJPG')
result = cv.VideoWriter(output_path, fourcc, 10, (width, height))

if not result.isOpened():
    print("❌ ERROR: Cannot create result video. Avoid OneDrive folder.")
    sys.exit(1)

# ----------------------------
#  Other parameters
# ----------------------------
g = geocoder.ip('me')
result_folder = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4

frame_counter = 0
save_index = 0
last_save_time = 0

# ----------------------------
#  Detection Loop
# ----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        print("Video finished.")
        break

    frame_counter += 1

    # Try detection
    try:
        classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    except cv.error:
        print("⚠ CUDA error → switching to CPU")
        net1.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        model1 = cv.dnn_DetectionModel(net1)
        model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)
        classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)

    # Draw detections
    for (classid, score, box) in zip(classes, scores, boxes):

        if score < 0.7:
            continue

        x, y, w, h = box
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, f"{round(score*100,1)}% pothole",
                   (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save detection every 2 seconds
        if time.time() - last_save_time >= 2:
            img_path = os.path.join(result_folder, f"pothole{save_index}.jpg")
            txt_path = os.path.join(result_folder, f"pothole{save_index}.txt")

            cv.imwrite(img_path, frame)
            with open(txt_path, "w") as f:
                f.write(str(g.latlng))

            save_index += 1
            last_save_time = time.time()

    # FPS counter
    elapsed = time.time() - starting_time
    fps = frame_counter / elapsed
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show + Save
    cv.imshow("frame", frame)
    result.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
#  End
# ----------------------------
cap.release()
result.release()
cv.destroyAllWindows()
print("✓ Processing complete.")
