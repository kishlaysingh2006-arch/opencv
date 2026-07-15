import cv2
import mediapipe as mp
import urllib.request
import os
import time

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading the new Tasks API face mesh model... (this only happens once)")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
start_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    frame_timestamp_ms = int((time.time() - start_time) * 1000)
    
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if result.face_landmarks:
        for face in result.face_landmarks:
            h, w, _ = image.shape
            
            for landmark in face:
                x_pixel = int(landmark.x * w)
                y_pixel = int(landmark.y * h)
                
                cv2.circle(image, (x_pixel, y_pixel), 1, (0, 255, 0), -1)

    cv2.imshow('Basic Face Mesh Tracker', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
