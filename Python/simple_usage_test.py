import cv2 as cv
import numpy as np

from PIL import Image

import mahalanobis_transformation

img1 = cv.imread("DOG.jpg") #reading images
img2 = cv.imread("brain (1).jpg")
#img2 = cv.imread("brain (1).jpg")
#img2 = cv.imread("brain (1).jpg")


ret1, img1 = cv.threshold(img1, 127, 255, 0) # pixels > 127 becomes 255 to make transparent background white, in more common case you can
                                          # map grey-scale-images to black-white
ret2, img2 = cv.threshold(img2, 127, 255, 0)

img1 = np.where(img1 == 255, 1, img1) #all 255 becomes 1
img2 = np.where(img2 == 255, 1, img2)

trans1 = np.array([[1, 0], [0, 1]]) #matrix induces Euclidean metric (you can use any symmetric positive-definite matrix)
trans2 = np.array([[2, 3], [3, 8]])
trans3 = np.array([[3, 0], [0, 5]])
lambda1, lambda2, theta = 2, 5, 0
#params: image, transformation matrix, connectivity_type, is_signed
#connectivity_type: 8-connectivity and 4-connectivity for 2d images, 26-connectivity and 6-connectivity for 3d


transformed1 = mahalanobis_transformation.MDT_connectivity(img1, trans1, "8-connectivity", 0)
transformed2 = mahalanobis_transformation.MDT_brute(img2, trans2, 1)
transformed3 = mahalanobis_transformation.MDT_window(img2, trans3, 10)
transformed4 = mahalanobis_transformation.MDT_ellipse(img1, lambda1, lambda2, theta, "4-connectivity", 0)

#brute algo: params same without connectivity
im1 = Image.fromarray(transformed1)
im2 = Image.fromarray(10 * transformed2)
im3 = Image.fromarray(10 * transformed3)
im4 = Image.fromarray(transformed4)


#im1.show()
#im2.show()
#im3.show()
#im4.show()