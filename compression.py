"""
Vector Quantization Image Compression.
Herleeyandi Markoni
11/27/2017
"""

import cv2
import lbg
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_training(img, block):
    """
    This function will generate the training codevector from the image via non-overlapped patch.
    """
    train_vec = []
    x = block[0]
    y = block[1]
    for i in range(0, img.shape[0], x):
        for j in range(0, img.shape[1], y):
            train_vec.append(img[i:i + x, j:j + y].reshape((x * y)))
    return (np.array(train_vec))

def generate_multi_training(path_list, block):
    """
    This function will generate the training codevector from the multi-image via non-overlapped patch.
    """
    img_list = []
    for path in path_list:
        img_list.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    train_vec = []
    x = block[0]
    y = block[1]
    for img in img_list:
        for i in range(0, img.shape[0], x):
            for j in range(0, img.shape[1], y):
                train_vec.append(img[i:i + x, j:j + y].reshape((x * y)))
    return (np.array(train_vec))


def distance(a, b):
    """
    This function will calculate the distance (MSE) of two vectors.
    """
    return np.mean((np.subtract(a, b) ** 2))


def closest_match(src, cb):
    """
    This function will get the closest distance (nearest) of the compared vectors.
    """
    c = np.zeros((cb.shape[0],))
    for i in range(0, cb.shape[0]):
        c[i] = distance(src, cb[i])
    minimum = np.argmin(c, axis=0)
    return minimum


def encode_image(img, cb, block):
    """
    This function will encode (compress) the image by sending the image block, vectorize it then get the index of the 
    closest vector to form the compressed data.
    """
    x = block[0]
    y = block[1]
    compressed = np.zeros((img.shape[0] // y, img.shape[1] // x))
    ix = 0
    for i in range(0, img.shape[0], x):
        iy = 0
        for j in range(0, img.shape[1], y):
            src = img[i:i + x, j:j + y].reshape((x * y)).copy()
            k = closest_match(src, cb)
            compressed[ix, iy] = k
            iy += 1
        ix += 1
    return compressed


def decode_image(cb, compressed, block):
    """
    This function will decode the compressed data beck to the image by taking the index of the codebook then copy the associate vector to the image block.
    """
    x = block[0]
    y = block[1]
    original = np.zeros((compressed.shape[0] * y, compressed.shape[1] * x))
    ix = 0
    for i in range(0, compressed.shape[0]):
        iy = 0
        for j in range(0, compressed.shape[1]):
            original[ix:ix + x, iy:iy + y] = cb[int(compressed[i, j])].reshape(block)
            iy += y
        ix += x
    return original


def save_weight(filename, cb):
    """
    This function will save the absolute and relative weight as CSV file.
    """
    fd = open(filename, 'a')
    for i in range(0, cb.shape[0]):
        linecsv = str(cb[i]) + '\n'
        fd.write(linecsv)
    fd.close()


def save_codebook(filename, cb):
    """
    This function will save the codebook as CSV file.
    """
    fd = open(filename, 'a')
    for i in range(0, cb.shape[0]):
        linecsv = ''
        for j in range(0, cb.shape[1]):
            linecsv = linecsv + str(cb[i, j]) + ','
        linecsv = linecsv + '\n'
        fd.write(linecsv)
    fd.close()


def save_csv(root, csv, cb, cb_abs_w, cb_rel_w):
    """
    This function will save the codebook and weight as CSV file given the associate name.
    """
    numpy_cb = np.array(cb)
    numpy_abs_w = np.array(cb_abs_w)
    numpy_rel_w = np.array(cb_rel_w)
    save_codebook(root + 'CB_' + csv + '.csv', numpy_cb)
    save_weight(root + '3CB_abs_' + csv + '.csv', numpy_abs_w)
    save_weight(root + '3CB_rel_' + csv + '.csv', numpy_rel_w)


def sim_protocol(img, cb_size, epsilon, block, root, outpng):
    """
    This function needod for doing simulation for different scenario.
    """
    train_X = generate_training(img, block)
    cb, cb_abs_w, cb_rel_w = lbg.generate_codebook(train_X, cb_size, epsilon)
    cb_n = np.array(cb)
    cb_abs_w_n = np.array(cb_abs_w)
    cb_rel_w_n = np.array(cb_rel_w)
    result = encode_image(img, cb_n, block)
    final_result = decode_image(cb_n, result, block)
    fig = plt.gcf()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    plt.imshow(final_result, cmap='gray')
    cv2.imwrite(root + outpng + '.png', final_result)
    save_csv(root, outpng, cb_n, cb_abs_w_n, cb_rel_w_n)

def sim_multi_protocol(path_list, cb_size, epsilon, block, root, outpng):
    """
    This function needod for doing simulation for different scenario.
    """
    train_X = generate_multi_training(path_list, block)
    cb, cb_abs_w, cb_rel_w = lbg.generate_codebook(train_X, cb_size, epsilon)
    cb_n = np.array(cb)
    cb_abs_w_n = np.array(cb_abs_w)
    cb_rel_w_n = np.array(cb_rel_w)
    save_csv(root, outpng, cb_n, cb_abs_w_n, cb_rel_w_n)
    print('Weight Saved as: '+outpng)

def sim_testing_protocol(inpath_list, weight, block, outpng):
    """
    This function needod for doing simulation for different scenario.
    """
    fig, ax = plt.subplots(nrows=1, ncols=4)
    idx = 1
    for inpath in inpath_list:
        img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
        cb = pd.read_csv(weight, header=None).as_matrix().astype('int')
        cb = cb[:,0:cb.shape[1]-1]
        result = encode_image(img, cb, block)
        final_result = decode_image(cb, result, block)
        rem = inpath.replace('./images/', '')
        cv2.imwrite(outpng + rem.replace('.csv',''), final_result)
        psnr_value = psnr(img, final_result)
        ax = plt.subplot(1, 4, idx)
        ax.set_title('PSNR = {}'.format(psnr_value))
        ax.imshow(final_result, cmap='gray')
        idx+=1
    fig.set_figheight(6)
    fig.set_figwidth(24)
    plt.show()


def psnr(img1, img2):
    """
    This function will calculate the PSNR of two images.
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def measure_psnr(apath, bpath):
    """
    This function will doing PSNR comparison of two images.
    """
    img1 = cv2.imread(apath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(bpath, cv2.IMREAD_GRAYSCALE)
    print('PSNR: {}'.format(psnr(img1, img2)))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.imshow(img1, cmap='gray')

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Result")
    ax2.imshow(img2, cmap='gray')

    fig.set_figheight(7)
    fig.set_figwidth(14)
    plt.show()
