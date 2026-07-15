import cv2
import mediapipe as mp
import urllib.request
import os
import time

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading the new Tasks API hand tracking model... (this only happens once)")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup the Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

landmarker = HandLandmarker.create_from_options(options)

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

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = image.shape
            
            pixel_points = []
            for landmark in hand:
                pixel_points.append((int(landmark.x * w), int(landmark.y * h)))
            
            for connection in HAND_CONNECTIONS:
                start_point = pixel_points[connection[0]]
                end_point = pixel_points[connection[1]]
                cv2.line(image, start_point, end_point, (0, 0, 255), 2) 
                
            for point in pixel_points:
                cv2.circle(image, point, 4, (0, 0, 255), -1)

    cv2.imshow('Tasks API Tracker', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
