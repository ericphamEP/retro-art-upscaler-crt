"""
Eric Pham
101104095

This file includes test code for applying CRT filters and resulting image cleanup.
There are 3 different methods, mainly for the cleanup image after CRT filter is applied.
"""
import math
import numpy as np
import cv2

import sys
np.set_printoptions(threshold=sys.maxsize)

# Instruction 1: Set below path to valid directory to store result images
OUTPUT_PATH = r'H:\Downloads\School\Year5\comp4102\project\examples\output\\'

# Instruction 2: Set below path to valid image file (minimum resolution pixel sprite)
INPUT_IMAGE = cv2.imread(r'H:\Downloads\School\Year5\comp4102\project\examples\input\street_fighter2.png')

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


def apply_scanlines(img, pixel_height, alternate=False, transparent=False):
    line_width = int(pixel_height * LINE_WIDTH_FACTOR)
    img_scanlines = img.copy()
    height, width, _ = img_scanlines.shape

    start = 0
    if alternate:
        start = int(pixel_height/2)
    for i in range(start, height+1, pixel_height):
        cv2.line(img_scanlines, (0, i), (width, i), (0, 0, 0), line_width)

    if transparent:
        tmp = cv2.cvtColor(img_scanlines, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(img_scanlines)
        img_scanlines = cv2.merge([b, g, r, alpha], 4)

    return img_scanlines


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
    kernel = np.repeat(minimal_kernel, scale, axis=0) # todo: try 24 for this as well
    kernel = np.repeat(kernel, scale, axis=1)
    kernel /= kernel.sum()/2

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
    scale = int(pixel_height)
    '''minimal_kernel = np.asarray([
        [0.1, 0.4, 1.0, 0.4, 0.1],
        [0.2, 0.6, 2.0, 0.6, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.6, 2.0, 0.6, 0.2],
        [0.1, 0.4, 1.0, 0.4, 0.1],
    ])
    kernel = np.repeat(minimal_kernel, scale, axis=0)
    kernel = np.repeat(kernel, scale, axis=1)
    kernel /= kernel.sum()'''
    half_scale = int(pixel_height/2)
    mid = int(pixel_height*3/2)
    kernel = np.zeros((pixel_height*3, pixel_height))
    kernel[mid+half_scale][half_scale] = 0.5
    kernel[mid-half_scale][half_scale] = 0.5
    print(kernel)

    img_rgb = cv2.cvtColor(img , cv2.COLOR_BGRA2BGR)
    img_filtered = cv2.filter2D(img_rgb, -1, kernel, borderType = cv2.BORDER_DEFAULT)
    return img_filtered


def overlay_transparent(img, background):
    foreground = img
    background = background
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    x_offset = (bg_w - fg_w) // 2
    y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    return background


def apply_sharpen(img, pixel_height):
    # Sharpen filter
    '''blur = cv2.GaussianBlur(img_combined, (0,0), sigmaX=40, sigmaY=40)
    result = cv2.addWeighted(img_combined, 2.0, blur, -1.0, 0)'''

    # Sharpen algo #2
    '''sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(result, -1, sharpen_filter)'''

    # Unsharp mask algo based on OpenCV docs: https://docs.opencv.org/4.x/d1/d10/classcv_1_1MatExpr.html#details
    sigma = 20
    threshold = 0.1
    amount = 1.0
    kernel_size=(pixel_height, pixel_height)
    blurred = cv2.GaussianBlur(img, (kernel_size), sigma)
    low_contrast_mask = np.absolute(img - blurred) < threshold
    sharpened = np.clip(img * (1.0 + amount) + blurred * (-amount), 0, 255).round().astype(np.uint8)
    np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


def art_upscale(img):
    img_upscaled = scale_image(img, SCALE_FACTOR, cv2.INTER_NEAREST)
    img_scanlines = apply_scanlines(img_upscaled, SCALE_FACTOR)

    # Method 1 - Use basic filters
    img_blur = apply_blur(img_scanlines, SCALE_FACTOR)
    #img_blur_test = apply_blur_test(img_scanlines, SCALE_FACTOR) #temp

    img_cleanup_filter = cleanup_filter(img_blur, SCALE_FACTOR)# temp img_blur
    img_cleanup = cv2.addWeighted(img_blur, 0.5, img_cleanup_filter, 0.5, 0)

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

    # Method 1.5 - Use transparent scanlines
    """img_blur_black = img_blur.copy()
    line_width = int(SCALE_FACTOR * LINE_WIDTH_FACTOR)
    height, width, _ = img_blur_black.shape
    for i in range(0, height+1, SCALE_FACTOR):
        cv2.line(img_blur_black, (0, i), (width, i), (0, 0, 0), line_width)
    
    tmp = cv2.cvtColor(img_blur_black, cv2.COLOR_BGRA2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r, _ = cv2.split(img_blur_black)
    img_blur_black = cv2.merge([b, g, r, alpha], 4)


    img_cleanup_transparent = overlay_transparent(img_blur_black, img_cleanup_filter)
    cv2.imshow('img cleanup transparent', scale_image(img_cleanup_transparent, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_cleanup_transparent.jpg', img_cleanup_transparent)"""

    cv2.imshow('img sharpen', scale_image(img_sharpen, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_sharpen.jpg', img_sharpen)

    # Method 2 - Use alternating scanlines and combine both
    """img_scanlines2 = apply_scanlines(img_upscaled, SCALE_FACTOR, alternate=True, transparent=True)
    img_blur2 = apply_blur(img_scanlines2, SCALE_FACTOR)
    img_combined = cv2.addWeighted(img_blur, 0.5, img_blur2, 0.5, 0)

    '''img_cleanup_filter2 = cleanup_filter(img_combined, SCALE_FACTOR/4)
    img_cleanup2 = cv2.addWeighted(img_combined, 0.42, img_cleanup_filter2, 0.98, 0)

    img_sharpen2 = apply_sharpen(img_cleanup2, SCALE_FACTOR)'''

    cv2.imshow('img scanlines 2', scale_image(img_scanlines2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_scanlines2.jpg', img_scanlines2)

    cv2.imshow('img crt2', scale_image(img_blur2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_crt2.jpg', img_blur2)

    cv2.imshow('img crt combo', scale_image(img_combined, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_crt_combo.jpg', img_combined)

    '''cv2.imshow('img crt combo cleanup', scale_image(img_cleanup2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_crt_combo_clean.jpg', img_cleanup2)

    cv2.imshow('img crt combo cleanup sharpen', scale_image(img_sharpen2, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_method2_sharpen.jpg', img_sharpen2)'''

    # test on combo
    img_combo_blur = cv2.GaussianBlur(img_combined, (SCALE_FACTOR, SCALE_FACTOR), sigmaX=13, sigmaY=15)
    cv2.imshow('combo blur test', scale_image(img_combo_blur, DOWNSCALE_FACTOR))
    img_combo_blur2 = cv2.GaussianBlur(img_combined, (int(2*SCALE_FACTOR/3)+1, int(2*SCALE_FACTOR/3)+1), sigmaX=15, sigmaY=31)
    cv2.imshow('combo blur test 2', scale_image(img_combo_blur2, DOWNSCALE_FACTOR))
    img_combo_blur_sharp = apply_sharpen(img_combo_blur, SCALE_FACTOR)
    cv2.imshow('combo blur test sharpp', scale_image(img_combo_blur_sharp, DOWNSCALE_FACTOR))"""

    blur_basic = cv2.GaussianBlur(img_upscaled, ((SCALE_FACTOR*2)-1, (SCALE_FACTOR*2)-1), sigmaX=15, sigmaY=15)
    cv2.imshow('basic blur to compare', scale_image(blur_basic, DOWNSCALE_FACTOR))
    cv2.imwrite(OUTPUT_PATH+'img_basic_blur.jpg', blur_basic)



if __name__ == '__main__':
    art_upscale(INPUT_IMAGE)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
