import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('bird1.jpg')

# smoothing
kernel = np.ones((7, 7), np.float32)/49
dst = cv.filter2D(img, -1, kernel)

# denoising
# dst = cv.medianBlur(dst, 9)
dst = cv.bilateralFilter(dst, 9, 750, 750)

# edge detection
dst_lapl = cv.Laplacian(dst, cv.CV_64F)

dst_can = cv.Canny(img, 100, 100)

# Sobel
grad_x = cv.Sobel(img, cv.CV_16S, 1, 0)
grad_y = cv.Sobel(img, cv.CV_16S, 0, 1)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

dst_sob = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# displaying image
plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(dst_lapl),plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(dst_can),plt.title('Canny')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(dst_sob),plt.title('Sobel')
plt.xticks([]), plt.yticks([])
plt.show()
