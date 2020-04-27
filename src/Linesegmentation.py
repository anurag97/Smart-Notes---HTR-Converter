import cv2
import numpy as np
import imutils

class LineSegmentation:
    def __init__(self,image):
        self.image = cv2.imread(image)
        
    def splitIntoLines(self):
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        #binary
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        #dilation
        kernel = np.ones((5,300), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        
        #find contours
        ctrs = cv2.findContours(img_dilation.copy(),  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ctrs = imutils.grab_contours(ctrs)

        #sort contours
        #sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        ctrs = ctrs[::-1] 

        Lines = []
        for i, ctr in enumerate(ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
        
            # Getting ROI
            roi = self.image[y:y+h, x:x+w]
        
            # show ROI
            Lines.append(roi)
          
            #plt.imshow(roi)
            #plt.show()
            #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
            #cv2.waitKey(0)
           
        return Lines