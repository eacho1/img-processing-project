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
    '''
        Perform convolution operation
        image: input image
        list_points: bitmask containing background pixels
        kernel: blur convolution kernel
    '''
    # dimensions of image
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    #"pad" the borders of the input image so the spacial size are not reduced after convolution
    pad = (kW -1)/2
    output = image.copy().astype(np.float32)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    #loop over masked areas of input image, "sliding" kernel across each (x,y)-coordinate       
    for point in list_points:
        x,y = point 
        x+=pad
        y+=pad
        #print(point) #debug
        if x in range(pad,iW + pad) and y in range (pad, iH + pad):
            #extract the ROI (region of interest) of image by extracting the "center" region of current (x,y)
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]

            #perform convolution by taking element-wise mulplicate of ROI and kernel, then summing matrix
            k = (roi*kernel).sum()

            output[y-pad, x-pad] = k

    #scale output image to be in range [0,255]
    output = rescale_intensity(output, in_range=(0,255))
    output = (output*255).astype("uint8")
    return output

def main():
    # parse all files in specified input images folder
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to input dataset of images")
    args = vars(ap.parse_args())
    output_file = "labels.png"
    for imagePath in glob.glob(args["images"] + "/*"):
        # load the image
        image = cv2.imread(imagePath)
        scale = 500.0/max(image.shape[:2])
        image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        cv2.imwrite("test.jpg", image)
        input_file = "test.jpg"
        # # Training model from https://goo.gl/ciEYZi
        saved_model_path = "crfrnn_keras_model.h5"

        model = get_crfrnn_model_def()
        model.load_weights(saved_model_path)
        # Trained weights

        # perform image segmentation
        img_data, img_h, img_w = util.get_preprocessed_image(input_file)
        probs = model.predict(img_data, verbose=False)[0, :, :, :]
        label_mask, segmentation = util.get_label_image(probs, img_h, img_w) 
        segmentation.save(output_file)

        image_array_rgb = np.array(Image.open(input_file))
        size = 5

        # box blur
        smallBlur = np.ones((size, size), dtype="float")*(1.0/(size*size))  

        # motion blur
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # Gaussian blur
        guassian_kernel_width = np.int32(16)
        guassian_kernel_half_width = np.int32(guassian_kernel_width/2)
        blur_sigma = np.float32(16)

        y,x = \
            scipy.mgrid[-guassian_kernel_half_width:guassian_kernel_half_width+1,
                    -guassian_kernel_half_width:guassian_kernel_half_width+1]
        blur_kernel_not_normalized = np.exp((-(x**2 + y**2))/(2 * blur_sigma**2))
        normalization_constant = np.float32(1) / np.sum(blur_kernel_not_normalized)
        gaussian_blur_kernel = (normalization_constant * blur_kernel_not_normalized).astype(np.float32)

        # split input image in r,g,b maps
        r_original,g_original,b_original = np.split(image_array_rgb, 3, axis=2)
        a_original = np.ones_like(r_original)*255

        rgba_original = np.concatenate((r_original,g_original,b_original), axis=2).copy()

        r    = r_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
        g    = g_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
        b    = b_original[1350:1650,2300:2600,0].copy().astype(np.uint8)
        a    = np.ones_like(r) * 255
        rgba = np.dstack((r,g,b,a)).copy()

        # choose blur kernel
        blur_kernel=gaussian_blur_kernel

        convolution_filter = blur_kernel

        # normalize label mask
        label_mask = np.divide(label_mask, 15)
        # zip and invert mask
        list_mask = zip(np.where(label_mask<1)[1], np.where(label_mask<1)[0])
        
        # perform convolution on r,g,b maps separately
        r_convolved = convolve(r_original, list_mask, convolution_filter)
        g_convolved = convolve(g_original, list_mask, convolution_filter)
        b_convolved = convolve(b_original, list_mask, convolution_filter)
        rgba_convolved = np.dstack((r_convolved, g_convolved, b_convolved)).copy()
        
        pyplot.imshow(rgba_convolved)
        pyplot.title("Convolved")
        pyplot.figure()
        pyplot.imshow(label_mask)
        pyplot.title("Label Mask")
        pyplot.figure()
        pyplot.imshow(blur_kernel, cmap="gray", interpolation="nearest")
        pyplot.title("blur_kernel")
        pyplot.figure()
        pyplot.imshow(rgba_original)
        pyplot.title("Original")
        pyplot.show()

if __name__ == "__main__":
    main()
