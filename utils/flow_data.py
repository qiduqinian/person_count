import math
import cv2

class FlowData(object):
    def __init__(self, interval = 5, fps=25):
        self.detection = []
        self.frames = interval*fps
        self.fps = fps
        self.interval = interval
    def update_flow(self, n, img):
        self.detection.append(n)
        if(len(self.detection) > self.frames):
            del(self.detection[0])
        people_avg = 0
        sz = round(len(self.detection)/self.fps)
        for i in range(sz):
            people_avg+=max(self.detection[i*self.fps:(i+1)*self.fps])
        if sz < self.interval:
            people_avg += sum(self.detection)/len(self.detection)
            sz+=1
        people_avg = math.floor(people_avg/sz)
        if people_avg <= 5:
            cong_status = "low"
        elif people_avg <=10:
            cong_status = "medium"
        else:
            cong_status = "high"
        cv2.putText(img, 'COUNT: '+str(people_avg), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, 'STATUS: '+cong_status, (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
