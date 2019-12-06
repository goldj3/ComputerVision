import numpy as np
import cv2

cap = cv2.VideoCapture('traffic.mp4')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1,frame2)   # taking the difference between two frames, may be negative so abs value is considered
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5,5), 0) # apply Gaussian blur to reduce sharp intensity changes among images, easier to treshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # simple thresholding to capture moving objects from still background
    dilated = cv2.dilate(thresh, None, iterations=5) # dilate image to enhance moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in contours:    # draw a rectangle around each contour around each moving vehicle if vehicle is greater than some area threshold
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 20000:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 5)
	    #cv2.drawContours(frame1,contours,-1, (0,255,0), 2)

    cv2.imshow("Frame", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    key = cv2.waitKey(1)
    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
