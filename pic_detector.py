import cv2


classifier = cv2.CascadeClassifier('myhaar.xml')
color = (0, 255, 0)
car_width=1.8

image=cv2.imread("pic/05.jpg")
height=image.shape[0]
width=image.shape[1]
print(height)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cars=classifier.detectMultiScale(gray,1.2,10,18,(24,24))
for (x,y,w,h) in cars:
    print(height-y)
    ppm = (float(car_width / w)*0.3+float(car_width/width)*0.7)
    dis=float(ppm*(height-y))
    print("dis=%.2f"%dis)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image,"%.2f m"%dis,(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

cv2.imwrite("result/05car2.jpg",image)
#cv2.imshow("Image",image)

cv2.destroyAllWindows()  # 释放窗口资源