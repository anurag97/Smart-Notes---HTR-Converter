from scannerClass import scannerClass
import matplotlib.pyplot as plt
import cv2

scanner = scannerClass()

#image path as parameter
image2 = scanner.getTransformedImage('')

plt.imshow(image2,cmap='gray')
plt.show()
