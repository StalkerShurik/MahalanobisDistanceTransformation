import cv2 as cv
import numpy as np

from PIL import Image

import mahalanobis_transformation

img = cv.imread("DOG.jpg") #reading images
#img_brain = cv.imread("brain (1).jpg")

ret, img = cv.threshold(img, 127, 255, 0) # pixels > 127 becomes 255 to make transparent background white, in more common case you can
                                          # map grey-scale-images to black-white

#ret_brain, img_brain = cv.threshold(img_brain, 127, 255, 0)

img = np.where(img == 255, 1, img) #all 255 becomes 1

#img_brain = np.where(img_brain == 255, 1, img_brain)

trans = np.array([[1, 0], [0, 1]]) #matrix induces Euclidean metric (you can use any symmetric positive-definite matrix)
#trans = np.array([[2, 3], [3, 8]])

#params: image, transformation matrix, connectivity_type, is_signed
#connectivity_type: 8-connectivity and 4-connectivity for 2d images, 26-connectivity and 6-connectivity for 3d
transformed = mahalanobis_transformation.MDT_connectivity(img, trans, "8-connectivity", 0)
#transformed_brain = mahalanobis_transformation.MDT_connectivity(img_brain, trans, "8-connectivity", 1)
#transformed_brain = mahalanobis_transformation.MDT_brute(img_brain, trans_euclidean, 1)
#transformed = mahalanobis_transformation.MDT_brute(img, trans, 0)
#brute algo: params same without connectivity
im = Image.fromarray(15 * transformed)


im.show()
