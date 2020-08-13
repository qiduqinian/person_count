import argparse
import time

import cv2
import torch
from numpy import random

from utils.datasets import LoadImages
from detector import build_detector
from utils.parser import get_config
from utils.general import plot_one_box
from utils.flow_data import FlowData

def detect(save_img=False):
    dataset = LoadImages(opt.source, img_size=cfg.YOLOV5.IMG_SIZE)
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        det = detector(img,im0s)
        flow_data.update_flow(len(det), im0s)
        for *xyxy, conf, cls in det:
            label = '%s %.2f' % ("people", conf)
            plot_one_box(xyxy, im0s, label=label, color=[203,192,255], line_thickness=1)
        cv2.imshow("test", im0s)
        cv2.waitKey(1)
        #vid_writer.write(im0s)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detector", type=str, default="./configs/yolov5.yaml")
    parser.add_argument('--source', type=str, default='data/People_sample_2.mp4', help='source')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    cfg = get_config()
    cfg.merge_from_file(opt.config_detector)
    cap = cv2.VideoCapture(opt.source)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    #vid_writer = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
    detector = build_detector(cfg, device=opt.device)
    flow_data = FlowData(interval=5, fps=fps)
    with torch.no_grad():
        detect()
