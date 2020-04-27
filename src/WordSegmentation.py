import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

class WordSegmentation():
    def __init__(self,img, kernelSize=25, sigma=11, theta=7, minArea=0):
        self.lineimg = img
        self.kernelSize=kernelSize
        self.sigma=sigma
        self.theta=theta
        self.minArea=minArea
        
    

    def wordSegmentation(self,lineimg):
        	# apply filter kernel
        	kernel = self.createKernel(self.kernelSize, self.sigma, self.theta)
        	imgFiltered = cv2.filter2D(lineimg, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
        	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        	imgThres = 255 - imgThres
        
        	# find connected components. OpenCV: return type differs between OpenCV2 and 3
        	if cv2.__version__.startswith('3.'):
        		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        	else:
        		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        	# append components to result
        	res = []
        	for c in components:
        		# skip small word candidates
        		if cv2.contourArea(c) < self.minArea:
        			continue
        		# append bounding box and image of word to result list
        		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        		(x, y, w, h) = currBox
        		currImg = lineimg[y:y+h, x:x+w]
        		res.append((currBox, currImg))
        
        	# return list of words, sorted by x-coordinate
        	return sorted(res, key=lambda entry:entry[0][0])
        

    def createKernel(self,kernelSize, sigma, theta):
        	"""create anisotropic filter kernel according to given parameters"""
        	assert kernelSize % 2 # must be odd size
        	halfSize = kernelSize // 2
        	
        	kernel = np.zeros([kernelSize, kernelSize])
        	sigmaX = sigma
        	sigmaY = sigma * theta
        	
        	for i in range(kernelSize):
        		for j in range(kernelSize):
        			x = i - halfSize
        			y = j - halfSize
        			
        			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
        			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
        			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
        			
        			kernel[i, j] = (xTerm + yTerm) * expTerm
        
        	kernel = kernel / np.sum(kernel)
        	return kernel
        
    def prepareImg(self,img, height):
        	"""convert given image to grayscale image (if needed) and resize to desired height"""
        	assert img.ndim in (2, 3)
        	if img.ndim == 3:
        		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        	h = img.shape[0]
        	factor = height / h
        	return cv2.resize(img, dsize=None, fx=factor, fy=factor)
        
    def segmentIntoWords(self):
        self.lineimg = self.prepareImg(self.lineimg, 50)
        res = self.wordSegmentation(self.lineimg)
        wordImages = []
        print('Segmented into %d words'%len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            
            #cv2.imwrite('../out/%s/%d.png'%(f, j), wordImg) # save word
            #cv2.rectangle(wordImg,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
            # output summary image with bounding boxes around words
            #wordImg = cv2.resize(wordImg,(32,128))
            
            #wordImg = np.reshape(wordImg,(128,32))
            #print("AAAAA",wordImg.shape)
            wordImages.append(wordImg)
         
        return wordImages