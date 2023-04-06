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
import numpy as np
import cv2

# Instruction 1: Set below path to valid directory to storee result images
OUTPUT_PATH = r'H:\Downloads\School\Year5\comp4102\project\examples\output\\'

# Instruction 2: Set below path to valid image file
INPUT_IMAGE = cv2.imread(r'H:\Downloads\School\Year5\comp4102\project\examples\input\streets_of_rage.jfif')


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


def apply_scanlines():
    pass


def brighten_image():
    # possibly optional?
    pass


def apply_light_bleed():
    # the whiter, the more bleed
    pass


def cleanup_filter_image():
    # include denoising, maybe sharpen?
    pass


def art_upscale(img):
    '''
    
    '''
    # call all other functions on this one
    result = img
    return result


if __name__ == '__main__':
    # test code
    img_resized, pixel_height = resize_image(INPUT_IMAGE)