import numpy as np
import scipy
import scipy.ndimage, scipy.ndimage.filters
from matplotlib import pyplot
from skimage.exposure import rescale_intensity
from PIL import Image
import cv2

def convolve(image, list_points, kernel):
    (iH, iW) = image.shape[:2]  #rows and columns
    (kH, kW) = kernel.shape[:2] #kernal rows and columns

    #"pad" the borders of the input image so the spacial size are not reduced after convolution
    pad = (kW -1)/2
    output = image.copy().astype(np.float32)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    #output = np.zeros((iH+pad, iW+pad), dtype="float32")

    #loop over input image, "sliding" kernel across each (x,y)-coordinate
    for point in list_points:
        x,y = point 
        x+=pad
        y+=pad
    #for y in np.arange(pad, iH+pad):
        #for x in np.arange(pad, iW + pad):
                #extract the ROI (region of interest) of image by extracting the "center" region of current (x,y)
        if x in range(pad,iW + pad) and y in range (pad, iH + pad):
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]

            #perform convolution by taking element-wise mulplicate of ROI and kernel, then summing matrix
            k = (roi*kernel).sum()

            output[y-pad, x-pad] = k

            #scale output image to be in range [0,255]
    output = rescale_intensity(output, in_range=(0,255))
    output = (output*255).astype("uint8")
    return output

smallBlur = np.ones((21,21), dtype="float")*(1.0/(21*21))


image = Image.open("test.jpg")
image_array_rgb = np.array(image)
r_original,g_original,b_original = np.split(image_array_rgb, 3, axis=2)
a_original = np.ones_like(r_original)*255

rgba_original = np.concatenate((r_original,g_original,b_original), axis=2).copy()
pyplot.subplot(121),pyplot.imshow(rgba_original);
pyplot.title("original")


r    = r_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
g    = g_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
b    = b_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
a    = np.ones_like(r) * 255
rgba = np.dstack((r,g,b,a)).copy()
#pyplot.imshow(rgba);
pyplot.title("rgba");

blur_kernel_width = np.int32(16)
blur_kernel_half_width = np.int32(8)
blur_sigma = np.float32(16)

y,x = \
        scipy.mgrid[-blur_kernel_half_width:blur_kernel_half_width+1,
                -blur_kernel_half_width:blur_kernel_half_width+1]
blur_kernel_not_normalized = np.exp((-(x**2 + y**2))/(2 * blur_sigma**2))
normalization_constant = np.float32(1) / np.sum(blur_kernel_not_normalized)
blur_kernel = (normalization_constant * blur_kernel_not_normalized).astype(np.float32)

#pyplot.imshow(blur_kernel, cmap="gray", interpolation="nearest")
pyplot.title("blur_kernel");

convolution_filter = blur_kernel

r_blurred= scipy.ndimage.filters.convolve(r, convolution_filter, mode="nearest")
g_blurred= scipy.ndimage.filters.convolve(g, convolution_filter, mode="nearest")
b_blurred= scipy.ndimage.filters.convolve(b, convolution_filter, mode="nearest")
rgba_blurred = np.dstack((r_blurred, g_blurred, b_blurred))

mask = np.ones((r_original.shape[0]-100,r_original.shape[1]-100))
list_mask = zip(np.where(mask==1)[0], np.where(mask==1)[1])
r_convolved = convolve(r_original, list_mask, convolution_filter)
g_convolved = convolve(g_original, list_mask, convolution_filter)
b_convolved = convolve(b_original, list_mask, convolution_filter)
rgba_convolved = np.dstack((r_convolved, g_convolved, b_convolved)).copy()
pyplot.subplot(122),pyplot.imshow(rgba_convolved)
#pyplot.subplot(122),pyplot.imshow(rgba_blurred)
pyplot.show()
