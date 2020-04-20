from scannerClass import scannerClass
import matplotlib.pyplot as plt
import cv2

scanner = scannerClass()
image2 = scanner.getTransformedImage('sample5.jpg')

print(image2.shape)

ret, thresh1 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY) 

plt.imshow(thresh1,cmap='gray')
plt.show()


cv2.imwrite('out.jpg',thresh1)