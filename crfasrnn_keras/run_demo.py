"""
MIT License
Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from crfrnn_model import get_crfrnn_model_def
import util
import cv2
import numpy as np
import scipy
import scipy.ndimage, scipy.ndimage.filters
import argparse
import glob
from matplotlib import pyplot
from skimage.exposure import rescale_intensity
from PIL import Image

def convolve(image, list_points, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    #"pad" the borders of the input image so the spacial size are not reduced after convolution
    pad = (kW -1)/2
    output = image.copy().astype(np.float32)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    #loop over input image, "sliding" kernel across each (x,y)-coordinate
    for point in list_points:
        x,y = point 
        x+=pad
        y+=pad
        print(point)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to input dataset of images")
    args = vars(ap.parse_args())
    output_file = "labels.png"
    for imagePath in glob.glob(args["images"] + "/*"):
        print("next")
        # load the image
        image = cv2.imread(imagePath)
        scale = 500.0/max(image.shape[:2])
        image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        cv2.imwrite("test.jpg", image)
        input_file = "test.jpg"
        # # Download the model from https://goo.gl/ciEYZi
        saved_model_path = "crfrnn_keras_model.h5"

        model = get_crfrnn_model_def()
        model.load_weights(saved_model_path)

        img_data, img_h, img_w = util.get_preprocessed_image(input_file)
        probs = model.predict(img_data, verbose=False)[0, :, :, :]
        label_mask, segmentation = util.get_label_image(probs, img_h, img_w) 
        segmentation.save(output_file)
        pyplot.imshow(label_mask)
        pyplot.title("Label Mask")
        pyplot.figure();
        print label_mask
        image_array_rgb = np.array(Image.open(input_file))
        smallBlur = np.ones((21,21), dtype="float")*(1.0/(21*21))
        r_original,g_original,b_original = np.split(image_array_rgb, 3, axis=2)
        a_original = np.ones_like(r_original)*255

        rgba_original = np.concatenate((r_original,g_original,b_original), axis=2).copy()

        r    = r_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
        g    = g_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
        b    = b_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
        a    = np.ones_like(r) * 255
        rgba = np.dstack((r,g,b,a)).copy()
        pyplot.imshow(rgba_original)
        pyplot.title("Original")
        pyplot.figure()

        blur_kernel_width = np.int32(16)
        blur_kernel_half_width = np.int32(8)
        blur_sigma = np.float32(16)

        y,x = \
            scipy.mgrid[-blur_kernel_half_width:blur_kernel_half_width+1,
                    -blur_kernel_half_width:blur_kernel_half_width+1]
        blur_kernel_not_normalized = np.exp((-(x**2 + y**2))/(2 * blur_sigma**2))
        normalization_constant = np.float32(1) / np.sum(blur_kernel_not_normalized)
        #blur_kernel = (normalization_constant * blur_kernel_not_normalized).astype(np.float32)
        blur_kernel=smallBlur

        pyplot.imshow(blur_kernel, cmap="gray", interpolation="nearest")
        pyplot.title("blur_kernel")
        pyplot.figure()

        convolution_filter = blur_kernel

        r_blurred= scipy.ndimage.filters.convolve(r, convolution_filter, mode="nearest")
        g_blurred= scipy.ndimage.filters.convolve(g, convolution_filter, mode="nearest")
        b_blurred= scipy.ndimage.filters.convolve(b, convolution_filter, mode="nearest")
        rgba_blurred = np.dstack((r_blurred, g_blurred, b_blurred))

        #label_mask = np.ones((r_original.shape[0]-100,r_original.shape[1]-100))
        label_mask = np.divide(label_mask, 15)
        #print label_mask
        list_mask = zip(np.where(label_mask<1)[1], np.where(label_mask<1)[0])
        #pyplot.subplot(122),pyplot.imshow(label_mask, cmap="gray", interpolation="nearest")
        #pyplot.title("label_mask");
        
        r_convolved = convolve(r_original, list_mask, convolution_filter)
        g_convolved = convolve(g_original, list_mask, convolution_filter)
        b_convolved = convolve(b_original, list_mask, convolution_filter)
        rgba_convolved = np.dstack((r_convolved, g_convolved, b_convolved)).copy()
        #pyplot.imshow("Original, Label, Convolved", np.hstack([rgba_original, label_mask, rgba_convolved]))
        
        pyplot.imshow(rgba_convolved)
        pyplot.title("Convolved")
        pyplot.figure()
        pyplot.show()

if __name__ == "__main__":
    main()
