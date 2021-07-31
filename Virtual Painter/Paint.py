# -*- coding: utf-8 -*-
"""
Created with 
    💗 
 by Sahil

"""


import cv2
import numpy as np
import os
import HandTrackingModule as htm

######################
brushThickness = 25
eraserThickness = 100
########################



folderPath = "Header"
myList = os.listdir(folderPath)  
print(myList)  

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]  

drawColor = (255, 0, 255) # default color


cap = cv2.VideoCapture(0)
cap.set(3, 1280) #width
cap.set(4, 720)  #height

detector = htm.handDetector(detectionCon=0.85,maxHands=1)
xp, yp = 0, 0  #origin
imgCanvas = np.zeros((720, 1280, 3), np.uint8) # canvas to draw

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # same side
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList)!=0:
        # print(lmList) #landmarks values

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]  #[8, 729, 356]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up 
        fingers = detector.fingersUp()
        # print(fingers)
        
        
        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # start from origin when hands detected again
            print("Selection Mode")
             # Checking for the click &   select image
            if y1 < 125:   # on header
                if 250 < x1 < 450: # first brush
                    header = overlayList[0]
                    drawColor = (255, 0, 255) #purple
                elif 550 < x1 < 750:  # second  brush
                    header = overlayList[1]
                    drawColor = (255, 0, 0) #blue
                elif 800 < x1 < 950:  # third brush
                    header = overlayList[2]
                    drawColor = (0, 255, 0) #green
                elif 1050 < x1 < 1200:  # eraser
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  #black
            #rectangle between index and middle finger        
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
         
        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            
            if xp == 0 and yp == 0:  #Do not draw from origin
                xp, yp = x1, y1 

            # draw line from previous points
            if drawColor == (0, 0, 0):  # increase the size of eraser
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)


            xp, yp = x1, y1 #previous points


        
    # image masking 
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) #grayimgae
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) #inverse binary image
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR) #colored image
    # adding images
    img = cv2.bitwise_and(img,imgInv) # black pen on original
    img = cv2.bitwise_or(img,imgCanvas) # colored pen on original


     # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0) # blend original image & canvas -> transparet image
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas) #black image with colored pen
    # cv2.imshow("Inv", imgInv) # white image with black pen
    # cv2.waitKey(1)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()