import torch
import numpy as np
import cv2

from numpy import random
from .models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def xyxy_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh

def xyxy_to_xywh_cut(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh

class YOLOv5(object):
    def __init__(self, weightfile, img_size=640, conf_thresh=0.5, nms_thresh=0.5, device=""):
        self.device = select_device(device)
        self.model = attempt_load(weightfile, map_location=self.device)
        self.imgsz = check_img_size(img_size, s=self.model.stride.max())
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        self.conf_thres = conf_thresh
        self.iou_thres = nms_thresh
        _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once

    def __call__(self, img, im0s, path=""):
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0, agnostic=True)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        return det
                #bbox = det[:, :4]
                #bbox = xyxy_to_xywh_cut(bbox)
                #cls_conf = det[:, 4]
               # cls_ids = det[:, 5].long()
           # else:
             #   bbox = torch.FloatTensor([]).reshape([0, 4])
             #   cls_conf = torch.FloatTensor([])
          #     cls_ids = torch.LongTensor([])
       # return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()
                



        
