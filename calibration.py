import numpy as np
import cv2
import os

class CamCal:
    def __init__(self,cal_dir,nx=9,ny=6):
        self.cal_dir=cal_dir
        self.nx=nx
        self.ny=ny
        self.mtx=None
        self.dist=None
        self.is_cal=False
    
    def cal(self):
        obj_pts=[]
        img_pts=[]
        objp=np.zeros((self.ny*self.nx,3),np.float32)
        objp[:,:2]=np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        if not os.path.exists(self.cal_dir):
            raise FileNotFoundError(f"Dir {self.cal_dir} not found")

        cal_imgs=os.listdir(self.cal_dir)
        if not cal_imgs:
            raise FileNotFoundError(f"No imgs in {self.cal_dir}")

        good_imgs=0
        for img_name in cal_imgs:
            path=os.path.join(self.cal_dir,img_name)
            img=cv2.imread(path)
            if img is None:
                print(f"[WARN] Can't load {path}. Skip")
                continue

            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray=cv2.equalizeHist(gray)
            gray=cv2.GaussianBlur(gray,(5,5),0)

            flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK
            ret,corners=cv2.findChessboardCorners(gray,(self.nx,self.ny),flags=flags)

            if ret:
                criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
                corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                img_pts.append(corners)
                obj_pts.append(objp.copy())
                good_imgs+=1
                print(f"[INFO] Found corners in {img_name} ({good_imgs} imgs done)")
            else:
                print(f"[INFO] No corners in {img_name}. Skip")

        if not obj_pts or not img_pts:
            self._set_def_cal()
            print("[WARN] No valid cal imgs. Using default")
            return

        ret,self.mtx,self.dist,rvecs,tvecs=cv2.calibrateCamera(obj_pts,img_pts,gray.shape[::-1],None,None)
        self.is_cal=True
        print(f"[INFO] Calibrated with {len(obj_pts)} imgs")
    
    def _set_def_cal(self):
        self.mtx=np.array([[1000,0,640],[0,1000,360],[0,0,1]],dtype=np.float64)
        self.dist=np.array([0,0,0,0,0],dtype=np.float64)
        self.is_cal=False

    def fix_dist(self,img):
        if self.mtx is None or self.dist is None:
            raise ValueError("Not calibrated. Run cal() first")
        return cv2.undistort(img,self.mtx,self.dist,None,self.mtx)
