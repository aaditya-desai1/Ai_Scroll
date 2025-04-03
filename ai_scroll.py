import cv2
import numpy as np
import pyautogui
import time

# Screen settings
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False  # Disable the fail-safe to prevent issues

# Scrolling parameters
scroll_speed = 5
is_scrolling = False

# Previous state
prev_gesture = None

# Initialize webcam
cap = cv2.VideoCapture(0)

def detect_hand_gesture(frame):
    """
    Detects hand gestures (palm or fist) using contour analysis
    Returns: "palm", "fist", or "unknown"
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color detection (this might need adjustment for different skin tones)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "unknown", mask
    
    # Find the largest contour (assumed to be the hand)
    contour = max(contours, key=cv2.contourArea)
    
    # Skip if contour is too small (not a hand)
    if cv2.contourArea(contour) < 5000:
        return "unknown", mask
    
    # Calculate convex hull
    hull = cv2.convexHull(contour)
    
    # Get convexity defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
    
    # Count fingers (number of defects)
    finger_count = 0
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate angle between fingers
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            
            # If angle is less than 90 degrees, it's likely a finger
            if angle <= np.pi / 2:
                finger_count += 1
                # Draw circle at the fingertip
                cv2.circle(frame, far, 5, [0, 0, 255], -1)
    
    # Draw contours on the frame
    cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
    cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
    
    # Determine gesture based on finger count
    if finger_count >= 4:
        gesture = "palm"
    elif finger_count <= 1:
        gesture = "fist"
    else:
        gesture = "unknown"
    
    return gesture, mask

print("Starting AI Scroll - Show your palm to scroll down, make a fist to stop scrolling")
print("Position your hand clearly in the camera frame")
print("Press 'q' to quit")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture image from camera")
            break
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Detect hand gesture
        gesture, mask = detect_hand_gesture(frame)
        
        # Display detected gesture on the frame
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Act on gesture changes
        if gesture != prev_gesture:
            if gesture == "palm":
                is_scrolling = True
                print("Palm detected - Scrolling down")
            elif gesture == "fist":
                is_scrolling = False
                print("Fist detected - Stopped scrolling")
        
        prev_gesture = gesture
        
        # Apply scrolling if palm is detected
        if is_scrolling:
            pyautogui.scroll(-scroll_speed)
        
        # Display the mask and resulting frame
        cv2.imshow('Mask', mask)
        cv2.imshow('AI Scroll', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("AI Scroll stopped") 