import cv2
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip
import os
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from calibration import CamCal
from lane_detector import LaneDetect as LaneDetector
from image_processing import ImgProc as ImageProcessor
from utils import calc_dist
from utils import detect_cars

# Display a test frame after calibration
def display_test_frame(calibrator, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video for test frame display.")
        return
    
    ret, frame = cap.read()
    if ret:
        frame = calibrator.fix_dist(frame)
        frame = cv2.resize(frame, (1280, 720))
        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Test Frame After Calibration")
        plt.show()
    else:
        logging.error("Failed to read a frame for test frame display.")
    
    cap.release()

def main(video_path, chessboard_folder, frame_index=0):
    # Initialize lane detector and image processor
    lane_detector = LaneDetector()
    image_processor = ImageProcessor()
    
    # Load YOLO model for car detection
    model = None
    try:
        model = YOLO('weights/yolov8n.pt')
        logging.info("YOLO model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}. Proceeding with lane detection only.")
    
    # Calibrate the camera
    calibrator = CamCal(chessboard_folder)
    try:
        calibrator.cal()
        logging.info("Camera calibration completed successfully.")
    except FileNotFoundError as e:
        logging.error(f"Calibration failed: {e}. Using default calibration.")
        calibrator._set_def_cal()
    
    # Use VideoFileClip to extract a single frame
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        logging.error(f"Could not open video: {e}")
        return
    
    # Extract the frame at the specified index
    try:
        frame = clip.get_frame(frame_index)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV compatibility
    except Exception as e:
        logging.error(f"Failed to extract frame at index {frame_index}: {e}")
        clip.close()
        return
    
    # Define perspective transform points (hardcoded for 1280x720 resolution)
    src_points = np.float32([(200, 720), (600, 450), (680, 450), (1100, 720)])
    dst_points = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])
    
    # Preprocess the frame (resize and undistort)
    frame = cv2.resize(frame, (1280, 720))
    undistorted_frame = calibrator.fix_dist(frame)
    
    # Apply preprocessing to get the intermediate frames
    preprocessed_frame = image_processor.thresh_img(undistorted_frame)
    bin_warp, M_inv = image_processor.transform(preprocessed_frame, src_points, dst_points)
    
    # Process for lane detection
    lane_frame = lane_detector.proc_img(undistorted_frame, image_processor, src_points, dst_points)
    
    # Process for car detection
    final_frame = detect_cars(lane_frame, model) if model else lane_frame
    
    # Display the four frames for comparison
    plt.figure(figsize=(24, 6))
    
    # Original Frame
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Original Frame")
    plt.axis('off')
    
    # Intermediate Frame (Warped Binary for Line Detection)
    plt.subplot(1, 4, 2)
    plt.imshow(bin_warp, cmap='gray')
    plt.title("Warped for Line Detection")
    plt.axis('off')
    
    # Preprocessed Frame (Binary Thresholded)
    plt.subplot(1, 4, 3)
    plt.imshow(preprocessed_frame, cmap='gray')
    plt.title("Preprocessed Frame")
    plt.axis('off')
    
    # Final Frame
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    plt.title("Final Result")
    plt.axis('off')
    
    plt.show()
    
    # Cleanup
    clip.close()
    logging.info("Frames displayed and resources released.")

if __name__ == "__main__":
    video_path = r"C:\Users\nabil\Downloads\New folder\sine raise.mp4"
    chessboard_folder = r"C:\Users\nabil\Downloads\archive\roboflow\chess"
    main(video_path, chessboard_folder, frame_index=0)