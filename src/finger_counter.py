def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    
    for tip in finger_tips:
        if landmarks[tip]['y'] < landmarks[tip - 2]['y']:
            count += 1
    
    return count
