# Lane_line-detection

![Uploading Untitled video - Made with Clipchamp (4).gif…]()



 Lane and Car Detection in Sandstorm Conditions
An advanced computer vision system for autonomous driving assistance, capable of detecting lanes and vehicles even in adverse weather conditions like sandstorms.

 Methodology
This system integrates traditional computer vision techniques and deep learning using YOLOv8. It includes:

Camera Calibration: Using chessboard images to remove lens distortion.

Image Preprocessing:

Dehazing using CLAHE

Gamma correction

Adaptive binary thresholding

Contrast enhancement

Lane Detection:

Sliding window approach

Histogram analysis

Polynomial fitting with smoothing

Car Detection:

YOLOv8 (yolov8n.pt)

Distance estimation using bounding box width


![Uploading Untitled video - Made with Clipchamp (3).gif…]()


Visualization: Overlays lanes, curvature, center offset, car boxes, and labels in real time.

 Preprocessing Steps
Preprocessing is crucial for handling low visibility due to sandstorms. Techniques used:

dehaze(): CLAHE in LAB space to reduce haze

adjust_gamma(): Brightens image for clarity

enhance_contrast(): Reinforces edge visibility

adaptive_threshold(): Dynamic Sobel, HSV, and HLS-based thresholding

remove_noise(): Morphological opening (5x5 kernel)

warp(): Bird’s-eye view transform for lane detection


These improve detection of lanes and vehicles under challenging conditions.

 Execution Pipeline
python
Copy
Edit
main(video_path, chessboard_folder)
Initialize Classes: CameraCalibrator, ImageProcessor, LaneDetector, and YOLOv8

OpenCV Video Read: Loads and resizes frames

Calibration: Undistorts frames using chessboard parameters

Preprocessing: Applies gamma correction, dehazing, adaptive thresholding

Lane Detection: Polynomial fitting, curvature, offset estimation

Object Detection: Detects cars using YOLO, estimates distance

Visualization: Shows lane overlays, bounding boxes, and metrics

 Dataset
Chessboard Images
C:\Users\nabil\Downloads\archive\roboflow\chess
Used for calibration (9×6 grid)

los-anglos high road 
Sandstorm Video
_sand_storm_project_video.mp4
Used to test detection robustness in adverse weather

 Challenges & Solutions
Challenge	Solution	Techniques Used
Low Visibility	Dehazing + Gamma	cv2.LAB + CLAHE + gamma
Dynamic Lighting	Adaptive thresholding	Based on mean brightness
Unstable Lane Fit	Polynomial smoothing	MSE-based averaging
Distance Estimation	Heuristic	Focal length and box width

python
Copy
Edit
distance = (known_width * focal_length) / bbox_width

 Results
Lane Detection: Curvature, offset, and lane positions shown in real time

Car Detection: Accurate bounding boxes with distances (5–30m)

Preprocessing Impact: Dramatic improvement in sandstorm visibility

Performance: ~8–16 FPS on CPU


![Untitled video - Made with Clipchamp (2)](https://github.com/user-attachments/assets/141334f5-51c6-4cfa-bf86-4d984897f54c)


