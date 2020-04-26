from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

class scannerClass:
    inputImage = None
    transImage = None
    
    def preProcessImage(self):
        self.inputImage = imutils.resize(self.inputImage, height = 500)
        gray = cv2.cvtColor(self.inputImage, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
    
        return edged        
        
    def findBorder(self,image):
        cnts = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        
        for c in cnts:
            per = cv2.arcLength(c,True)
            approxPoly = cv2.approxPolyDP(c,0.01*per,closed = True)
            if(len(approxPoly) == 4):
                brd = approxPoly
                break
         
      
        #brd = cnts[perList.index(min(perList))]
        border = np.full((4,2),0,dtype = 'float32')
        for i in range(len(brd)):
            border[i] = brd[i][0]
            
        return border
        
    def getTransformedImage(self,path):
        self.inputImage = cv2.imread(path)
        edgedImage = self.preProcessImage()
        plt.imshow(edgedImage)
        plt.show()
        Border = self.findBorder(edgedImage)
            
        src = np.full((4,2),0,dtype = 'float32')
        #for source
        s = np.sum(Border,axis=1)
        
        src[0] = Border[np.argmin(s)]      #top left
        src[2] = Border[np.argmax(s)]      #bottom right
        
        
        diff = np.diff(Border,axis = 1)
        src[1] = Border[np.argmin(diff)]      #top right
        src[3] = Border[np.argmax(diff)]      #bottom left
        
        
        tl = src[0]
        br = src[2]
        
        tr = src[1]
        bl = src[3]
        
        #for destination
        
        #transformation
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dest = np.array([[0, 0],
                         [maxWidth - 1, 0],
                         [maxWidth - 1, maxHeight - 1],
                         [0, maxHeight - 1]],
                        dtype = "float32")
        
        M = cv2.getPerspectiveTransform(src, dest)
        
        warped = cv2.warpPerspective(self.inputImage, M, (maxWidth, maxHeight))
    
        '''
        T = threshold_local(warped, 11, offset = 10, method = "gaussian")
        warped = (warped > T).astype("uint8") * 25
        '''
        return warped