import cv2
import numpy as np
from moviepy.editor import VideoFileClip

class VideoProcessor:
    def __init__(self, src_points, dst_points):
        self.src_points = src_points
        self.dst_points = dst_points
        
    def process_video_cv2(self, input_path, output_path, lane_detector, image_processor):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open input video: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Failed to initialize VideoWriter for {output_path}")
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame, lane_detector, image_processor)
            out.write(processed_frame)
            
        cap.release()
        out.release()
        
    def process_video_moviepy(self, input_path, output_path, lane_detector, image_processor):
        clip = VideoFileClip(input_path)
        processed_clip = clip.fl_image(lambda frame: self.process_frame(frame, lane_detector, image_processor))
        processed_clip.write_videofile(
            output_path,
            audio=False,
            codec='libx264',
            preset='medium',
            ffmpeg_params=['-an', '-vf', 'format=yuv420p'],
            verbose=True,
            logger='bar'
        )
        
    def process_frame(self, frame, lane_detector, image_processor):
        if frame is None or frame.size == 0:
            return frame.copy() if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
            
        # Process image
        binary_thresh = image_processor.binary_thresholded(frame)
        binary_warped, M_inv = image_processor.warp(binary_thresh, self.src_points, self.dst_points)
        
        # Detect lanes
        if not lane_detector.left_fit_hist:
            leftx, lefty, rightx, righty = lane_detector.find_lane_pixels_using_histogram(binary_warped)
        else:
            leftx, lefty, rightx, righty = lane_detector.find_lane_pixels_using_prev_poly(binary_warped)
            if len(lefty) == 0 or len(righty) == 0:
                leftx, lefty, rightx, righty = lane_detector.find_lane_pixels_using_histogram(binary_warped)
                
        left_fit, right_fit, left_fitx, right_fitx, ploty = lane_detector.fit_poly(
            binary_warped, leftx, lefty, rightx, righty)
        lane_detector.update_fit_history(left_fit, right_fit)
        
        # Measure curvature and position
        left_curverad, right_curverad = lane_detector.measure_curvature_meters(
            binary_warped, left_fitx, right_fitx, ploty)
        veh_pos = lane_detector.measure_position_meters(binary_warped, left_fit, right_fit)
        
        # Project lane info
        out_img = lane_detector.project_lane_info(
            frame, binary_warped, ploty, left_fitx, right_fitx, M_inv, 
            left_curverad, right_curverad, veh_pos)
            
        return out_img