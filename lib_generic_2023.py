import numpy as np
import matplotlib.pyplot as plt  # Plot images and graphics
import cv2  # Open, save and split images
import time  # Measure of time
from math import log10, sqrt  # Basic Operators

# from functools import singledispatchmethod
# from sklearn import metrics
# from scipy import signal
# from scipy import ndimage as nd
# import pandas as pd
# from scipy import stats
# import inspect

# -----------------------------------


# Generic Functions

def open_img(path, /, mode=1, echo=False):
    """
    Load an image of type png, jpg o tif
    :param path: path of the image file
    :type path: <str>
    :param mode: 1 for png and jpg, -1 for tif
    :type mode: <int>
    :param echo: Returns an echo of the function execution
    :type echo: <bool>
    :return: an instance of <np.ndarray>
    """
    if echo:
        print('* Loading image (' + path + ')...')
    # read image
    img = cv2.imread(path, mode)
    if echo:
        print(' - Size: ', img.shape[0], ' x ', img.shape[1])
        print(' - Gray scale: min = ', np.amin(img), '. max = ', np.amax(img))
        print('')
    return img


def extract_sub_img(path, corner, sub_img_size, /, mode=0, echo=False):
    """
    Extract a sub-imagen of the imagen
    :param path: path of the image file
    :type path: <str>
    :param corner: the upper left corner coordinates of the sub-image
    :type corner: a list or a <np.ndarray>
    :param sub_img_size: the size of the sub-image, rows and cols
    :type corner: a list or a <np.ndarray>
    :param mode: 0 for png and jpg, -1 for tif
    :type mode: <int>
    :param echo: Returns an echo of the function execution
    :type echo: <bool>
    :return: a sub-image of the image,  <np.ndarray>
    """
    if echo:
        print('* Extracting sub-image of (' + path + ')...')
    # read image
    img = cv2.imread(path, mode)
    # extract the sub-image
    sub_img = img[corner[0]:corner[0] + sub_img_size[0], corner[1]:corner[1] + sub_img_size[1]]
    if echo:
        print(' - Size image: ', img.shape[0], ' x ', img.shape[1])
        print(' - Gray scale image: min = ', np.amin(img), '. max = ', np.amax(img))
        print(' - Size sub-image: ', sub_img.shape[0], ' x ', sub_img.shape[1])
        print(' - Gray scale sub-image: min = ', np.amin(img), '. max = ', np.amax(img))
        print('')
    return sub_img


def save_img(img, path, /, echo=False):
    if echo:
        print('* Saving (' + path + ')...', end=" ")
    img = normalize_img(img, '[0,255]')
    cv2.imwrite(path, img)
    if echo:
        print('saved. \n')


# building..
def show_image_cv2(vector_images, vector_names, title, cols_max=4, figsize=(10, 4)):
    """
    Function that shows 1 or many images in only one figure
    """
    num_images = len(vector_names)
    if num_images == 1:
        img_norm = normalize_img(vector_images[0], '[0,255]').astype(np.uint8)
        plt.figure(title)
        plt.imshow(img_norm, "gray")
        plt.title(vector_names[0])
        plt.show()
    if num_images > 1:
        # cols_img is the maximum number of columns of images in the figure
        rows_plots = ((num_images - 1) // cols_max) + 1
        if rows_plots > 1:
            cols_plots = cols_max
        else:
            cols_plots = num_images
        fig, axes = plt.subplots(rows_plots, cols_plots, figsize=figsize)
        ax = axes.ravel()
        for i in range(num_images):
            img_norm = normalize_img(vector_images[i], '[0,255]').astype(np.uint8)
            ax[i].imshow(img_norm, "gray")
            ax[i].set_title(vector_names[i])
        for i in range(rows_plots * cols_plots - num_images):
            ax[num_images + i].axis('off')
        fig.suptitle(title)
        plt.show()


# building..
def show_plot_cv2(vector_images, vector_names, title, cols_max=4, figsize=(10, 4), savefig=False, name_savefig='plot'):
    """
    Function that shows 1 or many plots in only one figure
    """
    num_images = len(vector_names)
    if num_images == 1:
        plt.figure(title)
        plt.plot(vector_images[0])
        plt.title(vector_names[0])
        plt.show()
    if num_images > 1:
        rows_plots = ((num_images - 1) // cols_max) + 1
        if rows_plots > 1:
            cols_plots = cols_max
        else:
            cols_plots = num_images
        fig, axes = plt.subplots(rows_plots, cols_plots, figsize=figsize)
        ax = axes.ravel()
        for i in range(num_images):
            ax[i].plot(vector_images[i])
            ax[i].set_title(vector_names[i])
        for i in range(rows_plots * cols_plots - num_images):
            ax[num_images + i].axis('off')
        fig.suptitle(title)
        if savefig:
            plt.savefig(name_savefig)
        plt.show()


def normalize_img(img, /, _range='[0,255]'):
    """
    Normalize the image in the interval [0,255] or [0,1]
    :param img: the imagen to normalize
    :type img: <np.ndarray>
    :param _range: [0,1] for unitary normalization, [0,255] for uint8 normalization
    :type _range: str
    """
    if _range == '[0,255]':
        return (img - np.amin(img)) * (255 / (np.amax(img) - np.amin(img)))
    elif _range == '[0,1]':
        return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


# building..
def get_pixel_range(img, echo=False):
    """
    Return the pixel range of the image
    """
    if echo:
        print('Range: min = ', np.amin(img), '. max = ', np.amax(img))
    return [np.amin(img), np.amax(img)]


# building..
def get_center(img, echo=False):
    """
    Return the center (int) of an imagen
    """
    r_mid = int(img.shape[0] / 2)
    c_mid = int(img.shape[1] / 2)
    if echo:
        print('center[x,y] = [ ', r_mid, ', ', c_mid, ' ]')
    return [r_mid, c_mid]


# building..
def psnr_k(img_original, img_processed):
    mse = np.mean((img_original - img_processed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        return 100  # Therefore, psnr have no importance.
    max_pixel = 255.0
    _psnr = 20 * log10(max_pixel / sqrt(mse))
    return _psnr


# building..
def time_measure(function):
    """Decorator to measure the execution time"""
    def measured_function(*args, **kwargs):
        start = time.time()
        c = function(*args, **kwargs)
        t = time.time() - start
        # print("Time: ", t)
        return c, t
    return measured_function
