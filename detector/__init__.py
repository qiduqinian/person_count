from .YOLOv5 import YOLOv5

__all__ = ['build_detector']

def build_detector(cfg,device):
    return YOLOv5(weightfile=cfg.YOLOV5.WEIGHT, img_size=cfg.YOLOV5.IMG_SIZE, 
                  conf_thresh=cfg.YOLOV5.SCORE_THRESH, nms_thresh= cfg.YOLOV5.NMS_THRESH, device=device)
