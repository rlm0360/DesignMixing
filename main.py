import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread("Data/moo1.JPEG")
img2 = cv2.imread("Data/moo_far.JPEG")

img1 = cv2.resize(img1,(480,360))
img2 = cv2.resize(img2,(480,360))

img1 = img1[80:300,0:480]
img2 = img2[80:300,0:480]

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch (descriptors1, descriptors2,k=2)

good_matches = []

for m1, m2 in matches:
  if m1.distance < 0.4*m2.distance:
    good_matches.append([m1])

window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (0, 0, 0)
thickness = 2


SIFT_matches =cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

image = cv2.putText(SIFT_matches, str(len(good_matches)), org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow("compare",image)

#cv2.imshow("test",img1_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()