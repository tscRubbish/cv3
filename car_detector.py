import cv2

def estimateDistance():
    pass


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

classifier = cv2.CascadeClassifier('myhaar.xml')
out=cv2.VideoWriter("result/out.avi",cv2.VideoWriter_fourcc('D','I','V','X'),fps,(frame_width,frame_height))
color = (0, 255, 0)
car_width=1.8

frameCounter=0
while True:
    ret, image = video.read()
    frameCounter+=1
    if type(image) == type(None):
        break
    if frameCounter>frames:
        break

    #if frameCounter%2==1:
    if True:
        height = image.shape[0]
        width = image.shape[1]
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cars=classifier.detectMultiScale(gray,1.2,10,18,(24,24))
        for (x,y,w,h) in cars:
            cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
            ppm = (float(car_width / w)+float(car_width/width))*0.5
            dis = float(ppm * (height - y) - 1)
            cv2.putText(image, "%.2f m" % dis, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(image)
    cv2.imshow("Image",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()  # 释放摄像头
out.release()
cv2.destroyAllWindows()  # 释放窗口资源

