import numpy as np
import cv2
import scipy.linalg

class OpticalFlowMotion(object):
    def __init__(self,):
        self.termcriteria =  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.feature_extract = cv2.FastFeatureDetector_create(30)
        self.prev_image = None
        self.prev_pts = None
        self.frame_id = -1
        self.w = 0.7
        self.use_score = True
        self.init_check = False
        
    def collect_pts_in_bbox(self, bbox, pts):
        x, y = pts[:, 0, 0], pts[:, 0, 1]
        xmin, ymin, xmax, ymax = bbox

        valid = (xmin < x) & (x < xmax) & (ymin < y) & (y < ymax)
        
        return valid
        # case1 = pts[:, 0, 0] > bbox[0]
        # case2 = pts[:, 0, 1] > bbox[1]
        # case3 = pts[:, 0, 0] < bbox[2]
        # case4 = pts[:, 0, 1] < bbox[3]
        # valid = case1 & case2 & case3 & case4
        
        # return valid
    
    def collect_pts_in_mask(self, mask, pts):
        pts = pts.astype(int)
        return mask[pts[:, 0, 1], pts[:, 0, 0]]
                

    def preprocessing(self, image_gray, mask):
        self.cur_pts = self.feature_extract.detect(image_gray, mask)
        if len(self.cur_pts) == 0:
            self.cur_pts = np.zeros((0, 1, 2), dtype=float)
        else:
            self.cur_pts = cv2.KeyPoint_convert(self.cur_pts)
            self.cur_pts = self.cur_pts.reshape(-1, 1, 2)

        if self.init_check and len(self.prev_pts) != 0:
            p2n_pts, status_p2n, err = cv2.calcOpticalFlowPyrLK(self.prev_image, image_gray, self.prev_pts, None, criteria=self.termcriteria)
            n2p_pts, status_n2p, err = cv2.calcOpticalFlowPyrLK(image_gray, self.prev_image, p2n_pts, None, criteria=self.termcriteria)

            valid1 = (status_p2n==1) & (status_n2p==1)   
            dpt = self.prev_pts-n2p_pts
            dpt = dpt**2
            dist = (dpt[:, :, 0] + dpt[:, :, 1])
            valid2 = dist < 2.25 #1.5픽셀 이하
            valid = valid1 & valid2
            self.prev_pts = self.prev_pts[valid].reshape(-1,1,2)
            self.p2n_pts = p2n_pts[valid].reshape(-1,1,2)

        self.init_check = True
        self.prev_image = image_gray

    def postprocessing(self):
        self.prev_pts = self.cur_pts

    def initiate(self, measurement):
        return measurement, True

    def predict(self, bbox, mask=None):
        valid_check = False
        if len(self.prev_pts) == 0:
            return bbox, valid_check
        if mask is None:
            box_pt_idx = self.collect_pts_in_bbox(bbox, self.prev_pts)
        else:
            box_pt_idx = self.collect_pts_in_mask(mask, self.prev_pts)
            
        pPt = self.prev_pts[box_pt_idx]
        nPt = self.p2n_pts[box_pt_idx]
        
        if len(pPt) != 0:
            
            px1 = pPt[:, 0, 0].min()
            py1 = pPt[:, 0, 1].min()
            px2 = pPt[:, 0, 0].max()
            py2 = pPt[:, 0, 1].max()
            pmx1 = bbox[0] - px1
            pmy1 = bbox[1] - py1
            pmx2 = bbox[2] - px2
            pmy2 = bbox[3] - py2

            nx1 = nPt[:, 0, 0].min()
            ny1 = nPt[:, 0, 1].min()
            nx2 = nPt[:, 0, 0].max()
            ny2 = nPt[:, 0, 1].max()

            valid_check = True
            predict_box = np.array([nx1 + pmx1, ny1 + pmy1, nx2 + pmx2, ny2 + pmy2], dtype=float)
        else:
            predict_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=float)

        return predict_box, valid_check

    def update(self, bbox, measurement):
        if self.use_score:
            score_w = (min(measurement[-1], 0.8) - 0.1) / 0.7
            p_w = (1-score_w) + self.w
            m_w = score_w + (1-self.w)
            s_w = p_w+m_w
            p_w = p_w/s_w
            m_w = m_w/s_w
        else:
            p_w = self.w
            m_w = (1-self.w)
        
        return p_w*bbox + m_w*measurement[:4]