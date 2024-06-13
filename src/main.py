import cv2
from hand_detector import HandDetector
from finger_counter import count_fingers

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, results = detector.find_hands(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = detector.get_landmarks(hand_landmarks)
                num_fingers = count_fingers(landmarks)
                
                cv2.putText(frame, f"Fingers: {num_fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Gesture Counter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
