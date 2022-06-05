import cv2
import numpy as np
import car_lane_detector

#设置视频路径
video=cv2.VideoCapture(r"vedio/06.mp4")
frame_width = int(video.get(3))
frame_height = int(video.get(4))
print("w:   "+str(frame_width))
print("h:   "+str(frame_height))
#帧率
fps = video.get(cv2.CAP_PROP_FPS)
# 总帧数(frames)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
print("帧数："+str(fps))
print("总帧数："+str(frames))
print("视屏总时长："+"{0:.2f}".format(frames/fps)+"秒")

out=cv2.VideoWriter("result/out2.avi",cv2.VideoWriter_fourcc('D','I','V','X'),fps,(frame_width,frame_height))
frameCounter=0
while True:
    ret, image = video.read()
    frameCounter+=1
    print(frameCounter)
    if type(image) == type(None):
        break
    if frameCounter>frames:
        break

    confThreshold = 0.25
    nmsThreshold = 0.45
    objThreshold = 0.5

    yolonet = car_lane_detector.yolo(confThreshold=confThreshold, nmsThreshold=nmsThreshold, objThreshold=objThreshold)
    outimg = yolonet.detect(image)
    out.write(outimg)
    #cv2.imshow("Title",outimg)

video.release()  # 释放摄像头
out.release()
cv2.destroyAllWindows()  # 释放窗口资源