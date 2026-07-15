import cv2
import mediapipe as mp
import math
import numpy as np
from synth_engine import HeadlessSynth
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class RadialMenu:
    def __init__(self, center, radius, options, color):
        self.center = center
        self.radius = radius
        self.options = options
        self.color = color
        self.num_slices = len(options)
        self.slice_angle = 360 / self.num_slices
        self.inner_radius = int(radius * 0.45)

    def get_selection(self, hand_x, hand_y):
        dist = math.hypot(hand_x - self.center[0], hand_y - self.center[1])
        
        if dist < self.inner_radius:
            return "OFF"
        
        if dist > self.radius * 1.7:
            return "OFF"

        angle_rad = math.atan2(hand_y - self.center[1], hand_x - self.center[0])
        angle_deg = math.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360
            
        adjusted_angle = (angle_deg + (self.slice_angle / 2)) % 360
        index = int(adjusted_angle // self.slice_angle)
        return self.options[index]

    def draw(self, frame, active_selection):
        cv2.circle(frame, self.center, self.radius, self.color, 2)
        cv2.circle(frame, self.center, self.inner_radius, self.color, 2)

        for i, option in enumerate(self.options):
            angle = math.radians(i * self.slice_angle)
            
            x2 = int(self.center[0] + self.radius * math.cos(angle))
            y2 = int(self.center[1] + self.radius * math.sin(angle))
            cv2.line(frame, self.center, (x2, y2), self.color, 1)

            text_angle = math.radians((i * self.slice_angle) + (self.slice_angle / 2))
            
            tx = int(self.center[0] + (self.radius * 0.65) * math.cos(text_angle))
            ty = int(self.center[1] + (self.radius * 0.65) * math.sin(text_angle))
            
            thickness = 3 if option == active_selection else 1
            text_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)[0]
            cv2.putText(frame, option, (tx - text_size[0]//2, ty + text_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, thickness)
            
        center_thickness = 3 if active_selection == "OFF" else 1
        off_size = cv2.getTextSize("OFF", cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_thickness)[0]
        cv2.putText(frame, "OFF", (self.center[0] - off_size[0]//2, self.center[1] + off_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, center_thickness)


def main():
    synth = HeadlessSynth()
    cap = cv2.VideoCapture(0)
    
    success, frame = cap.read()
    if not success:
        print("Failed to access webcam.")
        return
        
    h, w, c = frame.shape
    
    left_menu = RadialMenu((int(w * 0.25), int(h * 0.5)), 240, ['C', 'D', 'E', 'F', 'G', 'A', 'B'], (255, 200, 0)) 
    right_menu = RadialMenu((int(w * 0.75), int(h * 0.5)), 240, ['maj', 'maj7', '7', 'sus4', 'm', 'm7', 'dim', 'aug'], (0, 200, 255)) 

    current_note = "OFF"
    current_chord = "OFF"

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6)
    
    detector = vision.HandLandmarker.create_from_options(options)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(mp_image)

            note_detected = "OFF"
            chord_detected = "OFF"

            if detection_result.hand_landmarks:
                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    
                    index_tip = hand_landmarks[8]
                    px, py = int(index_tip.x * w), int(index_tip.y * h)
                    
                    cv2.circle(frame, (px, py), 10, (0, 255, 0), cv2.FILLED)

                    label = detection_result.handedness[i][0].category_name
                    
                    if label == "Right": 
                        note_detected = left_menu.get_selection(px, py)
                    elif label == "Left": 
                        chord_detected = right_menu.get_selection(px, py)

            if note_detected != current_note or chord_detected != current_chord:
                current_note = note_detected
                current_chord = chord_detected
                
                active_chord = "maj" if current_chord == "OFF" else current_chord
                
                if current_note != "OFF":
                    synth.set_chord(current_note, active_chord)
                else:
                    synth.set_chord("OFF", "")

            left_menu.draw(frame, current_note)
            right_menu.draw(frame, current_chord)

            status_text = f"Playing: {current_note} {current_chord}" if current_note != "OFF" else "Muted"
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Soundgo Clone", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        synth.stop()

if __name__ == "__main__":
    main()