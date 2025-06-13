import numpy as np
import cv2

class LaneDetect:
    def __init__(self,nwins=9,margin=150,minpix=50,smooth_win=20):
        self.nwins=nwins
        self.margin=margin
        self.minpix=minpix
        self.smooth_win=smooth_win
        self.lfit_hist=[]
        self.rfit_hist=[]
        self.prev_lfit=None
        self.prev_rfit=None

    def find_pixels_hist(self,bin_warp):
        hist=np.sum(bin_warp[bin_warp.shape[0]//2:,:],axis=0)
        mid=int(hist.shape[0]//2)
        lx_base=np.argmax(hist[:mid])
        rx_base=np.argmax(hist[mid:])+mid
        
        win_h=int(bin_warp.shape[0]//self.nwins)
        nonzero=bin_warp.nonzero()
        nzy=np.array(nonzero[0])
        nzx=np.array(nonzero[1])
        
        lx_curr=lx_base
        rx_curr=rx_base
        l_lane=[]
        r_lane=[]
        
        for win in range(self.nwins):
            y_low=bin_warp.shape[0]-(win+1)*win_h
            y_high=bin_warp.shape[0]-win*win_h
            xl_low=lx_curr-self.margin
            xl_high=lx_curr+self.margin
            xr_low=rx_curr-self.margin
            xr_high=rx_curr+self.margin
            
            good_l=((nzy>=y_low)&(nzy<y_high)&(nzx>=xl_low)&(nzx<xl_high)).nonzero()[0]
            good_r=((nzy>=y_low)&(nzy<y_high)&(nzx>=xr_low)&(nzx<xr_high)).nonzero()[0]
            
            l_lane.append(good_l)
            r_lane.append(good_r)
            
            if len(good_l)>self.minpix:
                lx_curr=int(np.mean(nzx[good_l]))
            if len(good_r)>self.minpix:
                rx_curr=int(np.mean(nzx[good_r]))
        
        try:
            l_lane=np.concatenate(l_lane)
            r_lane=np.concatenate(r_lane)
        except ValueError:
            pass
            
        lx=nzx[l_lane] if len(l_lane)>0 else np.array([])
        ly=nzy[l_lane] if len(l_lane)>0 else np.array([])
        rx=nzx[r_lane] if len(r_lane)>0 else np.array([])
        ry=nzy[r_lane] if len(r_lane)>0 else np.array([])
        
        return lx,ly,rx,ry
    
    def find_pixels_prev(self,bin_warp):
        if self.prev_lfit is None or self.prev_rfit is None:
            return self.find_pixels_hist(bin_warp)
            
        nonzero=bin_warp.nonzero()
        nzy=np.array(nonzero[0])
        nzx=np.array(nonzero[1])
        
        l_lane=((nzx>(self.prev_lfit[0]*(nzy**2)+self.prev_lfit[1]*nzy+
                self.prev_lfit[2]-self.margin))&
                (nzx<(self.prev_lfit[0]*(nzy**2)+
                self.prev_lfit[1]*nzy+self.prev_lfit[2]+self.margin))).nonzero()[0]
        
        r_lane=((nzx>(self.prev_rfit[0]*(nzy**2)+self.prev_rfit[1]*nzy+
                self.prev_rfit[2]-self.margin))&
                (nzx<(self.prev_rfit[0]*(nzy**2)+
                self.prev_rfit[1]*nzy+self.prev_rfit[2]+self.margin))).nonzero()[0]
        
        lx=nzx[l_lane]
        ly=nzy[l_lane]
        rx=nzx[r_lane]
        ry=nzy[r_lane]
        
        return lx,ly,rx,ry
    
    def calc_poly(self,bin_warp,lx,ly,rx,ry):
        l_mse=float('inf')
        if len(ly)>10 and len(lx)>0 and not np.any(np.isnan(lx)) and not np.any(np.isnan(ly)):
            lfit=np.polyfit(ly,lx,2)
            lfit_fn=np.poly1d(lfit)
            l_mse=np.mean((lfit_fn(ly)-lx)**2)
        else:
            lfit=self.prev_lfit if self.prev_lfit is not None else None

        r_mse=float('inf')
        if len(ry)>10 and len(rx)>0 and not np.any(np.isnan(rx)) and not np.any(np.isnan(ry)):
            rfit=np.polyfit(ry,rx,2)
            rfit_fn=np.poly1d(rfit)
            r_mse=np.mean((rfit_fn(ry)-rx)**2)
        else:
            rfit=self.prev_rfit if self.prev_rfit is not None else None

        ploty=np.linspace(0,bin_warp.shape[0]-1,bin_warp.shape[0])

        try:
            lfitx=lfit[0]*ploty**2+lfit[1]*ploty+lfit[2] if lfit is not None else np.full_like(ploty,bin_warp.shape[1]//4,dtype=np.float64)
            rfitx=rfit[0]*ploty**2+rfit[1]*ploty+rfit[2] if rfit is not None else np.full_like(ploty,3*bin_warp.shape[1]//4,dtype=np.float64)
        except (TypeError,AttributeError,ValueError):
            lfitx=np.full_like(ploty,bin_warp.shape[1]//4,dtype=np.float64)
            rfitx=np.full_like(ploty,3*bin_warp.shape[1]//4,dtype=np.float64)
            lfit=None
            rfit=None

        return lfit,rfit,l_mse,r_mse,lfitx,rfitx,ploty
    
    def upd_fit_hist(self,lfit,l_mse,rfit,r_mse):
        if lfit is not None:
            self.lfit_hist.append((lfit,l_mse))
            if len(self.lfit_hist)>self.smooth_win:
                self.lfit_hist.pop(0)
            valid_hist=[(fit,mse) for fit,mse in self.lfit_hist if mse<float('inf')]
            if len(valid_hist)>0:
                fits=np.array([fit for fit,mse in valid_hist])
                weights=np.array([1/(mse+1e-6) for _,mse in valid_hist])
                self.prev_lfit=np.average(fits,axis=0,weights=weights)
            else:
                self.prev_lfit=None

        if rfit is not None:
            self.rfit_hist.append((rfit,r_mse))
            if len(self.rfit_hist)>self.smooth_win:
                self.rfit_hist.pop(0)
            valid_hist=[(fit,mse) for fit,mse in self.rfit_hist if mse<float('inf')]
            if len(valid_hist)>0:
                fits=np.array([fit for fit,mse in valid_hist])
                weights=np.array([1/(mse+1e-6) for _,mse in valid_hist])
                self.prev_rfit=np.average(fits,axis=0,weights=weights)
            else:
                self.prev_rfit=None
    
    def calc_curve_m(self,bin_warp,lfitx,rfitx,ploty):
        ym_per_pix=30/720
        xm_per_pix=3.7/700
        
        lfit_cr=np.polyfit(ploty*ym_per_pix,lfitx*xm_per_pix,2)
        rfit_cr=np.polyfit(ploty*ym_per_pix,rfitx*xm_per_pix,2)
        
        y_eval=np.max(ploty)
        
        lcurve=float('inf') if np.abs(2*lfit_cr[0])<=1e-6 else (
            (1+(2*lfit_cr[0]*y_eval*ym_per_pix+lfit_cr[1])**2)**1.5)/np.absolute(2*lfit_cr[0])
        rcurve=float('inf') if np.abs(2*rfit_cr[0])<=1e-6 else (
            (1+(2*rfit_cr[0]*y_eval*ym_per_pix+rfit_cr[1])**2)**1.5)/np.absolute(2*rfit_cr[0])
            
        return lcurve,rcurve
    
    def calc_pos_m(self,bin_warp,lfit,rfit):
        xm_per_pix=3.7/700
        y_max=bin_warp.shape[0]
        
        if lfit is not None and rfit is not None:
            lx_pos=lfit[0]*y_max**2+lfit[1]*y_max+lfit[2]
            rx_pos=rfit[0]*y_max**2+rfit[1]*y_max+rfit[2]
            lane_ctr=(lx_pos+rx_pos)/2
        else:
            lane_ctr=bin_warp.shape[1]/2
            
        veh_pos=((bin_warp.shape[1]/2)-lane_ctr)*xm_per_pix
        return veh_pos if veh_pos is not None else 0.0
    
    def draw_lanes(self,bin_warp,lfitx,rfitx,ploty):
        out=np.dstack((bin_warp,bin_warp,bin_warp))*255
        win_img=np.zeros_like(out)
        
        l_win1=np.array([np.transpose(np.vstack([lfitx-self.margin,ploty]))])
        l_win2=np.array([np.flipud(np.transpose(np.vstack([lfitx+self.margin,ploty])))])
        l_pts=np.hstack((l_win1,l_win2))
        
        r_win1=np.array([np.transpose(np.vstack([rfitx-self.margin,ploty]))])
        r_win2=np.array([np.flipud(np.transpose(np.vstack([rfitx+self.margin,ploty])))])
        r_pts=np.hstack((r_win1,r_win2))
        
        cv2.fillPoly(win_img,np.int32([l_pts]),(100,100,0))
        cv2.fillPoly(win_img,np.int32([r_pts]),(100,100,0))
        
        return cv2.addWeighted(out,1,win_img,0.3,0)
    
    def show_lane_info(self,img,bin_warp,ploty,lfitx,rfitx,M_inv,lcurve,rcurve,veh_pos):
        warp_zero=np.zeros_like(bin_warp).astype(np.uint8)
        color_warp=np.dstack((warp_zero,warp_zero,warp_zero))
        
        pts_l=np.array([np.transpose(np.vstack([lfitx,ploty]))])
        pts_r=np.array([np.flipud(np.transpose(np.vstack([rfitx,ploty])))])
        pts=np.hstack((pts_l,pts_r))
        
        cv2.fillPoly(color_warp,np.int32([pts]),(0,255,0))
        new_warp=cv2.warpPerspective(color_warp,M_inv,(img.shape[1],img.shape[0]))
        out=cv2.addWeighted(img,1,new_warp,0.3,0)
        
        avg_curve=(lcurve+rcurve)/2 if np.isfinite(lcurve) and np.isfinite(rcurve) else 0.0
        
        cv2.putText(out,f'Curve Radius [m]: {avg_curve:.2f}',(40,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.6,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(out,f'Center Offset [m]: {veh_pos:.2f}',(40,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.6,(255,255,255),2,cv2.LINE_AA)
        
        return out

    def proc_img(self,img,img_proc,src_pts,dst_pts):
        bin_thresh=img_proc.thresh_img(img)
        bin_warp,M_inv=img_proc.transform(bin_thresh,src_pts,dst_pts)

        if self.prev_lfit is None or self.prev_rfit is None:
            lx,ly,rx,ry=self.find_pixels_hist(bin_warp)
        else:
            lx,ly,rx,ry=self.find_pixels_prev(bin_warp)

        lfit,rfit,l_mse,r_mse,lfitx,rfitx,ploty=self.calc_poly(bin_warp,lx,ly,rx,ry)
        
        self.upd_fit_hist(lfit,l_mse,rfit,r_mse)
        
        lcurve,rcurve=self.calc_curve_m(bin_warp,lfitx,rfitx,ploty)
        veh_pos=self.calc_pos_m(bin_warp,lfit,rfit)
        
        final=self.show_lane_info(img,bin_warp,ploty,lfitx,rfitx,M_inv,lcurve,rcurve,veh_pos)
        
        return final