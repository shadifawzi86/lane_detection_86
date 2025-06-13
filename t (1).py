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

def main(video_path, chessboard_folder):
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
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video.")
        return
    
    # Define perspective transform points (hardcoded for 1280x720 resolution)
    src_points = np.float32([(200, 720), (600, 450), (680, 450), (1100, 720)])
    dst_points = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])
    
    # Set up video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = r'C:\Users\nabil\Downloads\New folder\output_project_video.mp4'
    out = cv2.VideoWriter(out_path, fourcc, fps, (1280, 720))  # Adjusted resolution to match resize
    
    try:
        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("Reached end of video or failed to read frame.")
                break
            
            # Preprocess the frame
            frame = cv2.resize(frame, (1280, 720))
            frame = calibrator.fix_dist(frame)
            
            # Process for lane detection
            lane_frame = lane_detector.proc_img(frame, image_processor, src_points, dst_points)
            
            # Process for car detection
            final_frame = detect_cars(lane_frame, model) if model else lane_frame
            
            # Display the result
            cv2.imshow('Lane and Car Detection', final_frame)
            out.write(final_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Video processing terminated by user.")
                break
    
    except Exception as e:
        logging.error(f"An error occurred during video processing: {e}")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logging.info("Video capture released, video saved, and windows closed.")

if __name__ == "__main__":
    video_path = r"C:\Users\nabil\Downloads\foggy_video.mp4"
    chessboard_folder = r"C:\Users\nabil\Downloads\archive\roboflow\chess"
    main(video_path, chessboard_folder)