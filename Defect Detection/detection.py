import cv2
import numpy as np
import math

def detect_defect(image_path):
    # Load the image and turn it grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # Ring as white
    
    # Find the outer edge of the ring
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pick the outer contour (one with a hole inside)
    outer_contour = None
    for i, cont in enumerate(contours):
        if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1:
            outer_contour = cont
            break
    if outer_contour is None:
        return "Error: No ring detected"
    
    # Find the center using moments
    moments = cv2.moments(binary)
    if moments['m00'] == 0:
        return "Error: Invalid moments"
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    # Get the edge points
    points = outer_contour[:, 0, :]
    
    # Measure distances to center and find median radius
    distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    radius = np.median(distances)
    
    # Check deviations from the median
    deviations = distances - radius
    
    # Set a relative threshold
    thresh = 0.005 * radius  # Tweak this if defects look too big or small
    
    # Find the biggest positive and negative deviations
    max_pos = np.max(deviations)
    max_neg = np.min(deviations)
    
    # Decide what we’ve got
    if max_pos <= thresh and max_neg >= -thresh:
        classification = "Good"
        location = None
    elif max_neg < -thresh and max_pos <= thresh:
        classification = "Cut"
    elif max_pos > thresh and max_neg >= -thresh:
        classification = "Flash"
    else:
        classification = "Mixed (Cut and Flash)"
    
    # Pinpoint the defect if there is one
    if classification != "Good":
        if classification == "Cut" or "Mixed" in classification:
            defect_idx = np.argmin(deviations)
        else:
            defect_idx = np.argmax(deviations)
        defect_point = points[defect_idx]
        defect_dev = deviations[defect_idx]
        angle = math.degrees(math.atan2(defect_point[1] - cy, defect_point[0] - cx))
        location = f"Angle: {angle:.1f}°, Magnitude: {defect_dev:.2f} pixels, Position: {defect_point}"
    else:
        location = "No defect"
    
    # Draw it out (green circle for fit, red dot for defect)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, (cx, cy), int(radius), (0, 255, 0), 2)
    if location:
        cv2.circle(color_img, tuple(defect_point), 5, (0, 0, 255), -1)
    cv2.imwrite(f"output_{classification}.png", color_img)
    
    return f"Classification: {classification}, Location: {location}"

# Test the images
TEST_IMAGES = [
    "/kaggle/input/detection-circular-disc/Circular disc/good/good.png",
    "/kaggle/input/detection-circular-disc/Circular disc/flashes/flash1.png",
    "/kaggle/input/detection-circular-disc/Circular disc/cuts/cut1.png"
]

for img_path in TEST_IMAGES:
    result = detect_defect(img_path)
    print(result)
