import cv2
import numpy as np
import pyautogui
import time

# Screen settings
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False  # Disable the fail-safe to prevent issues

# Scrolling parameters
scroll_speed = 15
is_scrolling = False

# Previous state
prev_gesture = None

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define region of interest for hand detection (bottom right portion of the frame)
# This will help avoid detecting faces which are typically in the center/upper area
roi_enabled = False  # Set to False if you want to detect hands anywhere in the frame

def detect_hand_gesture(frame):
    """
    Detects hand gestures (palm or fist) using contour analysis
    Returns: "palm", "fist", or "unknown"
    """
    height, width, _ = frame.shape
    
    # Define region of interest (bottom right quadrant of the frame)
    if roi_enabled:
        roi_x = width // 2
        roi_y = height // 2
        roi_width = width // 2
        roi_height = height // 2
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
        cv2.putText(frame, "Hand detection area", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Extract ROI for processing
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        if roi.size == 0:  # Check if ROI is valid
            return "unknown", np.zeros_like(frame[:, :, 0])
    else:
        roi = frame
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
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
    
    # Create a full-frame mask for display
    if roi_enabled:
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = mask
        mask_for_display = full_mask
    else:
        mask_for_display = mask
    
    if not contours:
        return "unknown", mask_for_display
    
    # Find the largest contour (assumed to be the hand)
    contour = max(contours, key=cv2.contourArea)
    
    # Skip if contour is too small (not a hand)
    if cv2.contourArea(contour) < 5000:
        return "unknown", mask_for_display
    
    # Get bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate aspect ratio of the bounding rectangle
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Faces typically have aspect ratio close to 1 (square/round)
    # Hands typically have different aspect ratios depending on the gesture
    # Reject if it's likely a face
    if 0.8 < aspect_ratio < 1.2 and cv2.contourArea(contour) > 15000:
        return "unknown", mask_for_display
    
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
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) if (2 * b * c) > 0 else 0
            
            # If angle is less than 90 degrees, it's likely a finger
            if angle <= np.pi / 2:
                finger_count += 1
                # Draw circle at the fingertip
                if roi_enabled:
                    # Adjust coordinates to the full frame
                    adjusted_far = (far[0] + roi_x, far[1] + roi_y)
                    cv2.circle(frame, adjusted_far, 5, [0, 0, 255], -1)
                else:
                    cv2.circle(roi, far, 5, [0, 0, 255], -1)
    
    # Draw contours on the frame
    if roi_enabled:
        # Create an array for the contours and hull in the full frame context
        contour_full = contour.copy()
        contour_full[:, :, 0] += roi_x
        contour_full[:, :, 1] += roi_y
        
        hull_full = hull.copy()
        hull_full[:, :, 0] += roi_x
        hull_full[:, :, 1] += roi_y
        
        cv2.drawContours(frame, [contour_full], 0, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull_full], 0, (0, 0, 255), 2)
    else:
        cv2.drawContours(roi, [contour], 0, (0, 255, 0), 2)
        cv2.drawContours(roi, [hull], 0, (0, 0, 255), 2)
    
    # Determine gesture based on finger count
    if finger_count >= 4:
        gesture = "palm"
    elif finger_count <= 1:
        gesture = "fist"
    else:
        gesture = "unknown"
    
    return gesture, mask_for_display

print("Starting AI Scroll - Show your palm to scroll down, make a fist to stop scrolling")
print("Position your hand in the bottom right corner of the camera frame")
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