import cv2
import argparse
import numpy as np

class yolo():
    def __init__(self, confThreshold=0.25, nmsThreshold=0.5, objThreshold=0.45):
        self.classes = ['car']
        num_classes = len(self.classes)
        anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.generate_grid()
        self.net = cv2.dnn.readNet('yolop.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    #生成用于yolo识别的网格
    def generate_grid(self):
        self.grid = [np.zeros(1)] * self.nl
        self.length = []
        self.areas = []
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            self.length.append(int(self.na * h * w))
            self.areas.append(h*w)
            if self.grid[i].shape[2:4] != (h,w):
                xv, yv = np.meshgrid(np.arange(w), np.arange(h))
                self.grid[i] = np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


    def resize_image(self, srcimg):
        newh, neww = self.inpHeight, self.inpWidth
        img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww

    #归一化
    def _normalize(self, img):
        img = img.astype(np.float32) / 255.0
        self.mean=np.mean(img)
        self.std=np.std(img)
        img = (img - self.mean) / self.std
        return img

    def detect(self, srcimg):
        img, newh, neww = self.resize_image(srcimg)
        img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(img)

        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        outimg = srcimg.copy()

        #车道区域识别
        drive_area_mask = outs[1][:, 0:(self.inpHeight ), 0:(self.inpWidth )]
        seg_id = np.argmax(drive_area_mask, axis=0).astype(np.uint8)
        seg_id = cv2.resize(seg_id, (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_NEAREST)
        outimg[seg_id == 1] = [0, 255, 0]

        #车道线识别
        lane_line_mask = outs[2][:, 0:(self.inpHeight), 0:(self.inpWidth)]
        seg_id = np.argmax(lane_line_mask, axis=0).astype(np.uint8)
        seg_id = cv2.resize(seg_id, (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_NEAREST)
        outimg[seg_id == 1] = [255, 0, 0]

        det_out = outs[0]
        row_ind = 0
        for i in range(self.nl):
            det_out[row_ind:row_ind+self.length[i], 0:2] = (det_out[row_ind:row_ind+self.length[i], 0:2] * 2. - 0.5 + np.tile(self.grid[i],(self.na, 1))) * int(self.stride[i])
            det_out[row_ind:row_ind+self.length[i], 2:4] = (det_out[row_ind:row_ind+self.length[i], 2:4] * 2) ** 2 * np.repeat(self.anchor_grid[i], self.areas[i], axis=0)
            row_ind += self.length[i]

        frameHeight = outimg.shape[0]
        frameWidth = outimg.shape[1]
        ratioh, ratiow = frameHeight / newh, frameWidth / neww

        #车辆识别
        classIds = []
        confidences = []
        boxes = []
        for detection in det_out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int((detection[0]) * ratiow)
                center_y = int((detection[1]) * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence) * detection[4])
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            (left, top, width, height) = (box[0], box[1], box[2], box[3])
            cv2.rectangle(outimg, (left, top), (left + width, top + height), (0, 0, 255), thickness=2)
        return outimg

if __name__ == "__main__":
    imgpath="pic/05.jpg"
    #class confidence
    confThreshold=0.25
    #nms iou thresh
    nmsThreshold=0.45
    #object confidence
    objThreshold=0.5

    yolonet = yolo(confThreshold=confThreshold, nmsThreshold=nmsThreshold, objThreshold=objThreshold)
    srcimg = cv2.imread(imgpath)
    outimg = yolonet.detect(srcimg)

    winName = 'car and lane detector'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, outimg)
    cv2.imwrite("result/05lane1.jpg",outimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#@misc{2108.11250,
#Author = {Dong Wu and Manwen Liao and Weitian Zhang and Xinggang Wang},
#Title = {YOLOP: You Only Look Once for Panoptic Driving Perception},
#Year = {2021},
#Eprint = {arXiv:2108.11250},
#}