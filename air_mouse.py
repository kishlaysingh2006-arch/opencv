import cv2
import mediapipe as mp
import numpy as np
import time
import math
from pynput.mouse import Controller, Button
from screeninfo import get_monitors

class AirMouse:
    def __init__(self, model_path='hand_landmarker.task', smoothing_alpha=0.15, box_scale=0.25):
        self.mouse = Controller()
  
        monitor = get_monitors()[0]
        self.screen_w = monitor.width
        self.screen_h = monitor.height
        
        self.alpha = smoothing_alpha
        self.prev_x, self.prev_y = 0, 0
        self.first_frame = True
        
        self.is_clicking = False
        self.click_threshold = 0.05 
        
        self.box_scale = box_scale
        
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.start_time = time.time()

    def calculate_bounds(self, frame_w, frame_h):
        cx, cy = frame_w // 2, frame_h // 2
        bx = int(frame_w * self.box_scale)
        by = int(frame_h * self.box_scale)
        return cx - bx, cx + bx, cy - by, cy + by

    def map_and_smooth(self, raw_x, raw_y, bounds):
        x_min, x_max, y_min, y_max = bounds
        
        norm_x = np.clip((raw_x - x_min) / (x_max - x_min), 0.0, 1.0)
        norm_y = np.clip((raw_y - y_min) / (y_max - y_min), 0.0, 1.0)
        
        target_x = norm_x * self.screen_w
        target_y = norm_y * self.screen_h
        
        if self.first_frame:
            self.prev_x, self.prev_y = target_x, target_y
            self.first_frame = False
            
        smooth_x = self.alpha * target_x + (1 - self.alpha) * self.prev_x
        smooth_y = self.alpha * target_y + (1 - self.alpha) * self.prev_y
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        bounds = self.calculate_bounds(w, h)
        x_min, x_max, y_min, y_max = bounds
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            
            index_tip = hand[8]
            thumb_tip = hand[4]
            
            raw_x = int(index_tip.x * w)
            raw_y = int(index_tip.y * h)
            screen_x, screen_y = self.map_and_smooth(raw_x, raw_y, bounds)
            self.mouse.position = (screen_x, screen_y)
            
            pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            
            if pinch_dist < self.click_threshold:
                if not self.is_clicking:
                    self.mouse.press(Button.left)
                    self.is_clicking = True

                cv2.circle(frame, (raw_x, raw_y), 8, (0, 0, 255), -1)
            else:
                if self.is_clicking:
                    self.mouse.release(Button.left)
                    self.is_clicking = False
                # Visual feedback: Green for hovering
                cv2.circle(frame, (raw_x, raw_y), 8, (0, 255, 0), -1)
                
        return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    air_mouse_system = AirMouse(model_path='hand_landmarker.task', smoothing_alpha=0.15, box_scale=0.25)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        output_frame = air_mouse_system.process_frame(frame)
        
        cv2.imshow("Air Mouse Core System", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    