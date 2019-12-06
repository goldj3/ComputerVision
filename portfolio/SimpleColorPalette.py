import cv2 as cv
import numpy as np

def empty():
    pass


def main():
    colorBar = np.zeros((512,512,3), np.uint8) # setting window size for color palette
    name = 'RGB Color Palette'  
    cv.namedWindow(name)
    cv.createTrackbar('R', name, 0, 255, empty) # making color trackbars to modify rgb values on window
    cv.createTrackbar('G', name, 0, 255, empty)
    cv.createTrackbar('B', name, 0, 255, empty)

    while(True):
        cv.imshow(name, colorBar)
        if cv.waitKey(1) == 27:
            break
        
        red = cv.getTrackbarPos('R', name) # finds rgb values the user is currently on
        green = cv.getTrackbarPos('G', name)
        blue = cv.getTrackbarPos('B', name)
    
        colorBar[:] = [red,green,blue] # updates the window to the current rgb values on the trackbar
        # print(red,green,blue)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()

