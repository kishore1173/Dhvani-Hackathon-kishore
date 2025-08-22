# Circular Disc Defect Detection

This repository contains a Python script for detecting defects (cuts and flashes) in circular disc blade images using OpenCV. The solution is designed to handle variations in diameter and translations during image acquisition, making it robust for real-world applications with limited datasets.

## Overview
The algorithm processes grayscale images of circular discs, identifying defects such as cuts (indentations) and flashes (protrusions) on the outer edge. It uses geometric analysis instead of machine learning due to the constrained dataset (only one image per class: good, flash, and cut). The code includes visualization of results with a fitted green circle and red markers for defects.

## Features
- Detects defects in circular disc blades.
- Handles different diameters and translations.
- Classifies defects into "Good," "Cut," or "Flash."
- Localizes defects with angle and position information.
- Outputs visualized images for verification.

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd circular-disc-defect-detection

Ensure the test images are placed in the following structure (adjust paths as needed):

good/good.png
flashes/flash1.png
cuts/cut1.png


Run the script:
bashpython detect_defect.py

Check the output images (output_Good.png, output_Flash.png, output_Cut.png) for visualizations.

Algorithm Details
Flowchart
The process flows like this:

Load the image in grayscale.
Apply binary thresholding to isolate the ring as white.
Find the outer contour using hierarchy.
Calculate the center via moments (handles translations).
Compute the median radius from contour points (handles varying diameters).
Measure deviations of each point from the median radius.
Analyze deviations with a relative threshold (e.g., 0.5% of radius).
Classify as "Good," "Cut," or "Flash" based on deviation signs.
Localize defects by finding extreme deviation points and their angles.
End with visualization and reporting.

Basic Algorithm to Find Defects
The algorithm starts by loading a grayscale image and applying binary thresholding to highlight the ring. Contours are detected to isolate the outer edge, and the center is found using moments to account for shifts. The median radius is calculated from distances to contour points, making it resilient to size differences. Deviations from this radius are checked against a relative threshold (e.g., 0.005 * radius) to flag defects. This geometric approach is efficient (O(N) with N contour points) and avoids machine learning due to the tiny dataset (one image per class), where training would overfit.
Localizing the Defect
Defects are pinpointed by identifying the contour point with the maximum deviation (negative for cuts, positive for flashes). The angle is computed using atan2 (0° at positive x-axis, counterclockwise), and the magnitude and (x, y) position are recorded. For example, a result might be "Cut at 45°, 2.5 pixels, at (150, 200)." Multiple defects are listed if present, though the provided images show single defects.
Classify the Defect to Flashes and Cut Marks
Classification depends on deviation analysis:

If all deviations are within ±threshold, it’s "Good."
If there’s a significant negative deviation (< -threshold), it’s a "Cut."
If there’s a significant positive deviation (> threshold), it’s a "Flash."
If both occur, it’s "Mixed," but the dominant defect (by magnitude) is prioritized. The threshold is relative (e.g., 0.005 * radius) for scalability.

Test Images
The script uses the following images:

/kaggle/input/detection-circular-disc/Circular disc/good/good.png
/kaggle/input/detection-circular-disc/Circular disc/flashes/flash1.png
/kaggle/input/detection-circular-disc/Circular disc/cuts/cut1.png

Adjust the TEST_IMAGES list in the code to match your local paths.
Output

Console output: Classification and location (e.g., "Classification: Cut, Location: Angle: 45.0°, Magnitude: -2.50 pixels, Position: [150 200]").
Saved images: Visualizations with a green fitted circle and red defect markers.

Notes

The threshold (0.005 * radius) may need adjustment based on defect size or image resolution.
No external pretrained models are used due to dataset constraints; OpenCV’s geometric methods suffice.
Ensure images are in the correct format and path structure.

License
This project is for educational purposes. Feel free to modify and use it, but no formal license is attached.
Contact
For questions, reach out via the repository issues page.
text### Flowchart
The process flows like this:
- Load the image in grayscale.
- Apply binary thresholding to isolate the ring as white.
- Find the outer contour using hierarchy.
- Calculate the center via moments (handles translations).
- Compute the median radius from contour points (handles varying diameters).
- Measure deviations of each point from the median radius.
- Analyze deviations with a relative threshold (e.g., 0.5% of radius).
- Classify as "Good," "Cut," or "Flash" based on deviation signs.
- Localize defects by finding extreme deviation points and their angles.
- End with visualization and reporting.

### Basic Algorithm to Find Defects
The algorithm starts by loading a grayscale image and applying binary thresholding to highlight the ring. Contours are detected to isolate the outer edge, and the center is found using moments to account for shifts. The median radius is calculated from distances to contour points, making it resilient to size differences. Deviations from this radius are checked against a relative threshold (e.g., 0.005 * radius) to flag defects. This geometric approach is efficient (O(N) with N contour points) and avoids machine learning due to the tiny dataset (one image per class), where training would overfit.

### Localizing the Defect
Defects are pinpointed by identifying the contour point with the maximum deviation (negative for cuts, positive for flashes). The angle is computed using `atan2` (0° at positive x-axis, counterclockwise), and the magnitude and (x, y) position are recorded. For example, a result might be "Cut at 45°, 2.5 pixels, at (150, 200)." Multiple defects are listed if present, though the provided images show single defects.

### Classify the Defect to Flashes and Cut Marks
Classification depends on deviation analysis:
- If all deviations are within ±threshold, it’s "Good."
- If there’s a significant negative deviation (< -threshold), it’s a "Cut."
- If there’s a significant positive deviation (> threshold), it’s a "Flash."
- If both occur, it’s "Mixed," but the dominant defect (by magnitude) is prioritized. The threshold is relative (e.g., 0.005 * radius) for scalability.
