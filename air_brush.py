import cv2
import mediapipe as mp
import numpy as np
import time
import math

class Stroke:
    """A data structure representing a single continuous brush stroke."""
    def __init__(self, color=(255, 0, 0), thickness=8):
        self.points = []
        self.color = color
        self.thickness = thickness

    def add_point(self, point):
        self.points.append(point)

class AirBrush:
    def __init__(self, model_path='hand_landmarker.task', smoothing_alpha=0.4, box_scale=0.5):

        self.alpha = smoothing_alpha
        self.prev_x, self.prev_y = 0, 0
        self.first_frame = True
        

        self.is_drawing = False
        self.draw_threshold = 0.05
        
        self.strokes = []          
        self.current_stroke = None 
        self.current_color = (255, 0, 0) 
        
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

    def map_and_smooth(self, raw_x, raw_y, bounds, frame_w, frame_h):
        """Maps wrist movement to the full application window."""
        x_min, x_max, y_min, y_max = bounds
        
        norm_x = np.clip((raw_x - x_min) / (x_max - x_min), 0.0, 1.0)
        norm_y = np.clip((raw_y - y_min) / (y_max - y_min), 0.0, 1.0)
        
        target_x = norm_x * frame_w
        target_y = norm_y * frame_h
        
        if self.first_frame:
            self.prev_x, self.prev_y = target_x, target_y
            self.first_frame = False
            
        smooth_x = self.alpha * target_x + (1 - self.alpha) * self.prev_x
        smooth_y = self.alpha * target_y + (1 - self.alpha) * self.prev_y
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def draw_canvas(self, frame):
        """Renders all historical strokes and the current active stroke."""
        all_strokes = self.strokes.copy()
        if self.current_stroke:
            all_strokes.append(self.current_stroke)
            
        for stroke in all_strokes:

            if len(stroke.points) > 1:
                for i in range(1, len(stroke.points)):
                    cv2.line(frame, stroke.points[i-1], stroke.points[i], 
                             stroke.color, stroke.thickness, cv2.LINE_AA)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        bounds = self.calculate_bounds(w, h)
        x_min, x_max, y_min, y_max = bounds
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (200, 200, 200), 1)
        
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
            
            brush_x, brush_y = self.map_and_smooth(raw_x, raw_y, bounds, w, h)
            pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            
            if pinch_dist < self.draw_threshold:
                if not self.is_drawing:

                    self.is_drawing = True
                    self.current_stroke = Stroke(color=self.current_color)
                
                self.current_stroke.add_point((brush_x, brush_y))

                cv2.circle(frame, (brush_x, brush_y), 10, self.current_color, -1)
            else:
                if self.is_drawing:

                    self.is_drawing = False
                    if self.current_stroke and len(self.current_stroke.points) > 1:
                        self.strokes.append(self.current_stroke)
                    self.current_stroke = None
                
                cv2.circle(frame, (brush_x, brush_y), 10, (255, 255, 255), 2)
        
        self.draw_canvas(frame)
        
        cv2.putText(frame, "Pinch to Draw | 'u': Undo | 'c': Clear | 'r','g','b': Colors", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        return frame

# Execution runtime
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    app = AirBrush(model_path='hand_landmarker.task')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        output_frame = app.process_frame(frame)
        
        cv2.imshow("AirBrush Canvas", output_frame)
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'): 
            if app.strokes:
                app.strokes.pop()
        elif key == ord('c'): 
            app.strokes.clear()
        elif key == ord('b'):
            app.current_color = (255, 0, 0)
        elif key == ord('g'):
            app.current_color = (0, 255, 0)
        elif key == ord('r'):
            app.current_color = (0, 0, 255)
            
    cap.release()
    cv2.destroyAllWindows()