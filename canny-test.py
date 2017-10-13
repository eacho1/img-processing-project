from matplotlib import pyplot as plt
import numpy as np
import argparse
import glob
import cv2

def auto_canny(image, sigma=0.33):
        #algorithm: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input dataset of images")
args = vars(ap.parse_args())

# loop over the images
for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# load the image, convert it to grayscale, and blur it slightly
	original_image = cv2.imread(imagePath)
        image = cv2.resize(original_image, (0,0), fx=0.1, fy=0.1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	wide = cv2.Canny(blurred, 10, 200)
	tight = cv2.Canny(blurred, 225, 250)
	auto = auto_canny(blurred,0.5)

	# show the images
        plt.subplot(122),plt.imshow(np.hstack([wide]),cmap='gray')
        plt.title('edges'),plt.xticks([]),plt.yticks([])
        plt.subplot(121),plt.imshow(auto,cmap='gray')
        plt.title('auto'),plt.xticks([]),plt.yticks([])
        plt.show()
#	cv2.imshow("Original", image)
#	cv2.imshow("Edges", np.hstack([auto]))
#	cv2.waitKey(0)
