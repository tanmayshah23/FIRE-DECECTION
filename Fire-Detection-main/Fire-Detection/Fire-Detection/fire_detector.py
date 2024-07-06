#used Convolutional Neural Networks (CNNs)
#camera vision(cv2)
import cv2
# To access xml file
fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml')

# Open the laptop camera (camera index 0) 
vid = cv2.VideoCapture(0)

while True:
    # Value in ret is True # To read video frame
    ret, frame = vid.read()
    
    # To convert frame into gray color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    
    # to provide frame resolution
    fires = fire_cascade.detectMultiScale(frame,scaleFactor=1.2, minNeighbors=5)

    ## to highlight fire with square
    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):#q is for quit
        break

# Release the camera and close all windows
vid.release()
cv2.destroyAllWindows()
