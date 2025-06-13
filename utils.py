import numpy as np
import cv2
import matplotlib.pyplot as plt

from moviepy import VideoFileClip

import os
import logging
import sys



def calc_dist(w,h):
    focal=1000
    known_w=2.0
    dist=(known_w*focal)/w
    return dist

def detect_cars(img,model):
    res=model(img)
    img_copy=img.copy()
    
    for r in res:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=box.conf[0]
            cls=int(box.cls[0])
            
            if model.names[cls]=='car' and conf>=0.5:
                cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,255),2)
                w=x2-x1
                dist=calc_dist(w,y2-y1)
                label=f'Car {conf:.2f}, Dist: {dist:.2f}m'
                cv2.putText(img_copy,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
    
    return img_copy

def test_frame(vid_path,idx,ld,ip,src_pts,dst_pts):
    clip=VideoFileClip(vid_path)
    frame=clip.get_frame(idx)
    
    bin_thresh=ip.thresh_img(frame)
    bin_warp,M_inv=ip.transform(bin_thresh,src_pts,dst_pts)
    
    lx,ly,rx,ry=ld.find_pixels_hist(bin_warp)
    lfit,rfit,l_mse,r_mse,lfitx,rfitx,ploty=ld.calc_poly(bin_warp,lx,ly,rx,ry)
    
    res_img=ld.draw_lanes(bin_warp,lfitx,rfitx,ploty)
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.title("Orig Frame")
    
    plt.subplot(1,2,2)
    plt.imshow(res_img)
    plt.plot(lfitx,ploty,color='green',label='Left Lane')
    plt.plot(rfitx,ploty,color='blue',label='Right Lane')
    plt.title("Warped with Fits")
    plt.legend()
    
    plt.show()
    
    return frame,bin_warp,lfitx,rfitx,ploty