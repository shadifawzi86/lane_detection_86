import numpy as np
import cv2





class ImgProc:
    @staticmethod
    def boost_contrast(img):
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        cl=clahe.apply(l)
        limg=cv2.merge((cl,a,b))
        return cv2.cvtColor(limg,cv2.COLOR_LAB2BGR)

    @staticmethod
    def clean_noise(bin_img):
        kernel=np.ones((5,5),np.uint8)
        return cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel)

    @staticmethod
    def thresh_img(img):
        img=ImgProc.boost_contrast(img)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        adapt_thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        
        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
        abs_sobelx=np.absolute(sobelx)
        scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        sx_bin=np.zeros_like(scaled_sobel)
        sx_bin[(scaled_sobel>=20)&(scaled_sobel<=255)]=1

        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        low_white=np.array([0,0,200])
        high_white=np.array([180,30,255])
        white_hsv=cv2.inRange(hsv,low_white,high_white)
        
        bin1=cv2.bitwise_or(sx_bin,white_hsv)
        
        hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        H=hls[:,:,0]
        S=hls[:,:,2]
        sat_bin=np.zeros_like(S)
        sat_bin[(S>90)&(S<=255)]=1
        hue_bin=np.zeros_like(H)
        hue_bin[(H>10)&(H<=25)]=1
        bin2=cv2.bitwise_or(hue_bin,sat_bin)
        
        binary=cv2.bitwise_or(bin1,bin2)
        mask=cv2.bitwise_and(adapt_thresh,adapt_thresh,mask=binary)
        binary=cv2.bitwise_or(mask,binary)
        
        return ImgProc.clean_noise(binary)

    @staticmethod
    def transform(img,src_pts,dst_pts):
        size=(img.shape[1],img.shape[0])
        M=cv2.getPerspectiveTransform(src_pts,dst_pts)
        M_inv=cv2.getPerspectiveTransform(dst_pts,src_pts)
        warped=cv2.warpPerspective(img,M,size)
        return warped,M_inv
    
    def warp(img, src_pts, dst_pts):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        warped = cv2.warpPerspective(img, M, img_size)
        return warped, M_inv


'''
class ImgProc:
    @staticmethod
    def transform(img, src_pts, dst_pts):
        # Alias for warp to match expected method name
        return ImgProc.warp(img, src_pts, dst_pts)

    @staticmethod
    def dehaze(img):
        # Enhance visibility in fog/rain using CLAHE on LAB channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def adjust_gamma(image, gamma=1.2):
        # Adjust gamma to improve visibility in low-light or foggy conditions
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def enhance_contrast(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def remove_noise(bin_img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def thresh_img(img):
        # Apply dehazing and gamma correction to handle fog/rain
        img = ImgProc.dehaze(img)
        img = ImgProc.adjust_gamma(img, gamma=1.2)
        img = ImgProc.enhance_contrast(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding based on image statistics
        mean_brightness = np.mean(gray_img)
        sobel_low = max(15, min(30, mean_brightness * 0.1))  # Adjust Sobel threshold dynamically
        sobel_high = min(255, max(200, mean_brightness * 1.5))
        adaptive_block_size = 11 if mean_brightness > 100 else 15  # Larger block for low brightness

        adaptive_thresh = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_block_size, 2
        )

        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= sobel_low) & (scaled_sobel <= sobel_high)] = 1

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Adjust HSV thresholds based on brightness
        white_value = max(180, min(220, mean_brightness * 0.9))
        lower_white_hsv = np.array([0, 0, white_value])
        upper_white_hsv = np.array([180, 30, 255])
        white_binary_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

        binary_1 = cv2.bitwise_or(sx_binary, white_binary_hsv)

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:, :, 0]
        S = hls[:, :, 2]
        # Adjust saturation threshold dynamically
        sat_low = max(70, min(90, mean_brightness * 0.5))
        sat_binary = np.zeros_like(S)
        sat_binary[(S > sat_low) & (S <= 255)] = 1
        hue_binary = np.zeros_like(H)
        hue_binary[(H > 10) & (H <= 25)] = 1
        binary_2 = cv2.bitwise_or(hue_binary, sat_binary)

        binary = cv2.bitwise_or(binary_1, binary_2)
        mask = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=binary)
        binary = cv2.bitwise_or(mask, binary)

        return ImgProc.remove_noise(binary)

    @staticmethod
    def warp(img, src_pts, dst_pts):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        warped = cv2.warpPerspective(img, M, img_size)
        return warped, M_inv
    
'''