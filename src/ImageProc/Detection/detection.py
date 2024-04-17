import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read in image
img= cv2.imread("Datasets\yes\Y6.jpg")
# print("width: {} pixels".format(img.shape[1]))
# print("height: {} pixels".format(img.shape[0]))
# print("channels: {}".format(img.shape[2]))
dim=(500,590)
img=cv2.resize(img, dim)
cv2.imshow("image", img)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY, 0.7)

# cv2.imshow("Brain with Skull",gray)

#Make a histogram of the intensities in the grayscale image
# plt.hist(gray.ravel(),256)
# plt.show()


#Threshold the image to binary using Otsu's method
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
# cv2.imshow('Applying Otsu',thresh)

colormask = np.zeros(img.shape, dtype=np.uint8)
colormask[thresh!=0] = np.array((0,0,255))
blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
# cv2.imshow('Blended', blended)


ret, markers = cv2.connectedComponents(thresh)
#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
#Get pixels which correspond to the brain
brain_mask = markers==largest_component
brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)
cv2.imshow('Connected Components',brain_out)

averaging = cv2.blur(brain_out,(11,11))

(T, thresh) = cv2.threshold(averaging, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh)

(T, threshInv) = cv2.threshold(averaging, 155, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshInv",threshInv)

cv2.waitKey(0)
cv2.destroyAllWindows()
