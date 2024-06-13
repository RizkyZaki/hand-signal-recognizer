import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame, self.results
    
    def get_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility,
            })
        return landmarks
