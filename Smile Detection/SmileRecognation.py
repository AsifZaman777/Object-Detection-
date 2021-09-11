import cv2

faceXML=cv2.CascadeClassifier("frontalface.xml")
smileXML=cv2.CascadeClassifier("smile.xml")

capture_camera=cv2.VideoCapture(0)
address="http://192.168.0.101:8080/video"
capture_camera.open(address)

while True:
    check,Capture=capture_camera.read() #reading from mobile camera
    grayScale=cv2.cvtColor(Capture,cv2.COLOR_BGR2GRAY)
    face=faceXML.detectMultiScale(grayScale,scaleFactor=1.1,minNeighbors=5)
    for x,y,width,height in face:
        image=cv2.rectangle(Capture,(x,y),(x+width,y+height),(255,0,0),3)
        smile = smileXML.detectMultiScale(grayScale, scaleFactor=1.8, minNeighbors=20)
        for x, y, width, height in smile:
            image = cv2.rectangle(Capture, (x, y), (x + width, y + height), (0, 255, 0), 3)

    cv2.imshow("Smile Detection",Capture)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       capture_camera.release()
       cv2.destroyAllWindows()

