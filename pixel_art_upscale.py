"""
Eric Pham
101104095
Denta Rogan
101112927
COMP 4102
Michael Genkin
April 12, 2023
Project - Code
"""
import math
import numpy as np
import cv2

# Instruction 1: Set below path to valid directory to storee result images
OUTPUT_PATH = r'H:\Downloads\School\Year5\comp4102\project\examples\output\\'

# Instruction 2: Set below path to valid image file (minimum resolution pixel sprite)
INPUT_IMAGE = cv2.imread(r'H:\Downloads\School\Year5\comp4102\project\examples\input\street_fighter2.png')
#INPUT_IMAGE = cv2.imread(r'H:\Downloads\School\Year5\comp4102\project\examples\input\upscaled\dracula.png')

# Constants
SCALE_FACTOR = 25
DOWNSCALE_FACTOR = 0.25
LINE_WIDTH_FACTOR = 0.4
BLUR_SIGMA = 20

# Note: Unused for now
"""
def pixel_value_distance(val1, val2):
    print(val1, val2, np.sqrt(sum((val1-val2)**2)))
    return np.sqrt(sum((val1-val2)**2))


def get_pixel_height(img):
    # find avg min vertical distance between colours "changing"
    rows, cols, _ = img.shape
    heights = []

    for j in range(cols):
        min_distance = rows
        curr_distance = 0
        curr_pixel_val = None
        for i in range(rows):
            val = img[i,j]
            if curr_pixel_val is None:
                curr_pixel_val = val
                continue

            if pixel_value_distance(val, curr_pixel_val) < 10:
                curr_distance += 1
            else:
                if curr_distance < min_distance:
                    min_distance = curr_distance
                curr_distance = 0
                curr_pixel_val = val
        heights.append(min_distance)
    
    return np.median(heights)

# Note: Unused for now
def resize_image(img):
    '''
    Find size of original artwork pixels in possibly-upscaled imag,
    and upscale to a resolution height of 2160 pixels (4k resolution).

    :input img: Image
    :return: resized image and height of artwork pixels
    '''
    scale = 2160 / img.shape[0]
    img_resized = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    pixel_height = get_pixel_height(img_resized)

    return img_resized, pixel_height
"""


def scale_image(img, scale, interpolation=None):
    '''
    Scale an image.

    :input img: Image
    :return: resized image and height of artwork pixels
    '''
    img_resized = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)
    return img_resized


def apply_scanlines(img, pixel_height, alternate=False):
    line_width = int(pixel_height * LINE_WIDTH_FACTOR)
    img_scanlines = img.copy()
    height, width, _ = img_scanlines.shape

    start = 0
    if alternate:
        start = int(pixel_height/2)
    for i in range(start, height+1, pixel_height):
        cv2.line(img_scanlines, (0, i), (width, i), (0, 0, 0), line_width)

    return img_scanlines


# todo: may not use
"""def apply_light_bleed(img, white_thresh, gain):
    '''
    Add bloom effect from white pixels.
    
    :input white_thresh: Threshold value to detect white
    :input gain: Bloom gain intensity
    '''
    # convert image to hsv colorspace as floats
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    _, s, v = cv2.split(hsv)

    # Desire low saturation and high brightness for white
    # Invert saturation and multiply with brightness
    sv = ((255-s) * v / 255).clip(0,255).astype(np.uint8)

    # Threshold white values
    thresh = cv2.inRange(hsv, (0, 0, white_thresh), (255, 50, 255))
    #thresh = cv2.threshold(sv, white_thresh, 255, cv2.THRESH_BINARY)[1]

    # Blur and make 3 channels
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA)
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # blend blur and image using gain on blur
    img_bloom = cv2.addWeighted(img, 1, blur, gain, 0)
    return img_bloom
"""


def apply_blur(img, pixel_height):
    scale = int(pixel_height/4)
    minimal_kernel = np.asarray([
        [2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0],
        [1.5, 3.5, 3.0, 3.0, 3.0, 3.5, 1.5],
        [1.5, 2.0, 3.5, 3.5, 3.5, 2.0, 1.5],
        [1.5, 2.0, 3.0, 4.0, 3.0, 2.0, 1.5],
        [1.5, 2.0, 3.5, 3.5, 3.5, 2.0, 1.5],
        [1.5, 3.5, 3.0, 3.0, 3.0, 3.5, 1.5],
        [2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0],
    ])

    # Expand minimal kernel to full size of pixel_height
    kernel = np.repeat(minimal_kernel, scale/2, axis=0) # todo: try 24 for this as well
    kernel = np.repeat(kernel, scale, axis=1)
    kernel /= kernel.sum()

    img_filtered = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_DEFAULT)
    return img_filtered


# Does not work
def apply_blur_test(img, pixel_height):
    minimal_kernel = np.asarray([
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        [1.5, 3.0, 3.0, 3.0, 3.0, 3.0, 1.5],
        [1.5, 2.0, 3.5, 3.5, 3.5, 2.0, 1.5],
        [1.5, 2.0, 3.0, 4.0, 3.0, 2.0, 1.5],
        [1.5, 2.0, 3.5, 3.5, 3.5, 2.0, 1.5],
        [1.5, 3.0, 3.0, 3.0, 3.0, 3.0, 1.5],
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    ])

    # Expand minimal kernel to full size of pixel_height
    kernel = np.repeat(minimal_kernel, 12, axis=0) # todo: try 24 for this as well
    kernel = np.repeat(kernel, 24, axis=1)
    kernel /= kernel.sum()

    v_kernel = np.repeat(minimal_kernel, 24, axis=0)
    v_kernel = np.repeat(v_kernel, 36, axis=1)
    v_kernel /= v_kernel.sum()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    h, s, v = cv2.split(hsv)

    h_filtered = cv2.filter2D(h, -1, kernel, borderType = cv2.BORDER_DEFAULT)
    s_filtered = cv2.filter2D(s, -1, kernel, borderType = cv2.BORDER_DEFAULT)
    v_filtered = cv2.filter2D(v, -1, v_kernel, borderType = cv2.BORDER_DEFAULT)
    merged = cv2.merge([h_filtered, s_filtered, v_filtered])
    img_blur = cv2.cvtColor(cv2.normalize(merged, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_HSV2BGR)
    return img_blur


def cleanup_filter(img, pixel_height):
    # include denoising, maybe sharpen?
    scale = int(pixel_height/3)
    minimal_kernel = np.asarray([
        [1.5, 1.8, 2.0, 1.8, 1.5],
        [1.8, 1.9, 2.0, 1.9, 1.8],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.8, 1.9, 2.0, 1.9, 1.8],
        [1.5, 1.8, 2.0, 1.8, 1.5],
    ])
    kernel = np.repeat(minimal_kernel, scale, axis=0)
    kernel = np.repeat(kernel, scale, axis=1)
    kernel /= kernel.sum()

    img_filtered = cv2.filter2D(img, -1, kernel, borderType = cv2.BORDER_DEFAULT)
    return img_filtered


def apply_sharpen(img, pixel_height):
    # Sharpen filter
    '''blur = cv2.GaussianBlur(img_combined, (0,0), sigmaX=40, sigmaY=40)
    result = cv2.addWeighted(img_combined, 2.0, blur, -1.0, 0)'''

    # Sharpen algo #2
    '''sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, sharpen_filter)'''

    # Unsharp mask algo based on OpenCV docs: https://docs.opencv.org/4.x/d1/d10/classcv_1_1MatExpr.html#details
    sigma = 20
    threshold = 5
    amount = 1.0
    kernel_size=(pixel_height*3, pixel_height*3)
    blurred = cv2.GaussianBlur(img, (kernel_size), sigma)
    low_contrast_mask = np.absolute(img - blurred) < threshold
    sharpened = np.clip(img * (1.0 + amount) + blurred * (-amount), 0, 255).round().astype(np.uint8)
    np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


def art_upscale(img):
    img_upscaled = scale_image(img, SCALE_FACTOR, cv2.INTER_NEAREST)
    img_scanlines = apply_scanlines(img_upscaled, SCALE_FACTOR)

    img_blur = apply_blur(img_scanlines, SCALE_FACTOR)
    #img_blur_test = apply_blur_test(img_scanlines, SCALE_FACTOR) #temp

    img_cleanup_filter = cleanup_filter(img_blur, SCALE_FACTOR)
    img_cleanup = cv2.addWeighted(img_blur, 0.42, img_cleanup_filter, 0.98, 0)

    img_sharpen = apply_sharpen(img_cleanup, SCALE_FACTOR)

    cv2.imshow('img original', scale_image(img_upscaled, DOWNSCALE_FACTOR))
    cv2.imshow('img scanlines', scale_image(img_scanlines, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_scanlines.jpg', img_scanlines)

    cv2.imshow('img crt', scale_image(img_blur, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_crt.jpg', img_blur)

    #cv2.imshow('img blur test', scale_image(img_blur_test, DOWNSCALE_FACTOR))

    cv2.imshow('img filtered clean', scale_image(img_cleanup_filter, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_clean_filter.jpg', img_cleanup_filter)

    cv2.imshow('img cleanup', scale_image(img_cleanup, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_cleanup.jpg', img_cleanup)

    cv2.imshow('img sharpen', scale_image(img_sharpen, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_sharpen.jpg', img_sharpen)

    # Method 2
    img_scanlines2 = apply_scanlines(img_upscaled, SCALE_FACTOR, alternate=True)
    img_blur2 = apply_blur(img_scanlines2, SCALE_FACTOR)
    img_combined = cv2.addWeighted(img_blur, 0.6, img_blur2, 0.6, 0)

    img_cleanup_filter2 = cleanup_filter(img_combined, SCALE_FACTOR)
    img_cleanup2 = cv2.addWeighted(img_combined, 0.42, img_cleanup_filter2, 0.98, 0)

    img_sharpen2 = apply_sharpen(img_cleanup2, SCALE_FACTOR)

    cv2.imshow('img scanlines 2', scale_image(img_scanlines2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_scanlines2.jpg', img_scanlines2)

    cv2.imshow('img crt2', scale_image(img_blur2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_crt2.jpg', img_blur2)

    cv2.imshow('img crt combo', scale_image(img_combined, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_crt_combo.jpg', img_combined)

    cv2.imshow('img crt combo cleanup', scale_image(img_cleanup2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_crt_combo_clean.jpg', img_cleanup2)

    cv2.imshow('img crt combo cleanup sharpen', scale_image(img_sharpen2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_sharpen.jpg', img_sharpen2)


if __name__ == '__main__':
    #INPUT_IMAGE = cv2.resize(INPUT_IMAGE, dsize=(0, 0), fx=1/14, fy=1/14) #temp downscale dracula

    art_upscale(INPUT_IMAGE)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
