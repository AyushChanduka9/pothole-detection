# real-time-pothole-detection
Pothole detection from real time video stream or images with Python.

YoloV4-Tiny and OpenCV used for this project.
- Clone the repository.
```
git clone https://github.com/noorkhokhar99/pothole-detection.git
```
- Goto the cloned folder.
```
cd pothole-detection

```

✅ 1. Install Python

Make sure you have Python 3.8 – 3.11 installed.

Check version:

python --version

✅ 2. Install OpenCV

OpenCV is installed through pip:

pip install opencv-python


If you need additional OpenCV contrib features:

pip install opencv-contrib-python

✅ 3. Install geocoder

pip install geocoder
pip install pillow


All set, we are good to go, now put your test image in root folder and name it as test.jpg
i.e, pothole-detection/test.jpg

Simolarly, put your test video in root folder and name it as tets.mp4
i.e, pothole-detection/test.mp4


To test the image file, run
python image.py

To test the vidoe file, run
python camera_video.py


To run the web app
python app.py