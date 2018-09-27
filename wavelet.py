import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import logging
import math
# import pywt

def read_image(img_name):
    img = plt.imread(img_name).astype(float)
    logging.debug(img.shape)
    return img

def show_img(img, file_name = None):
    plt.imshow(img, cmap = 'gray')
    if not file_name:
        plt.show()
    else:
        plt.savefig('output/' + file_name)

    
h = np.array([1 + math.sqrt(3), 3 + math.sqrt(3), 3 - math.sqrt(3), \
                1 - math.sqrt(3)]) / (4 * math.sqrt(2))
g = np.array([h[3], -h[2], h[1], -h[0]])

def D4_1D(input_list, compress = False):

    odd_padded = False
    if len(input_list) % 2 == 1:
        odd_padded = True

    # repeat the last pixel at the right boundary
    # note we make a copy of the input list
    # because we do NOT want to change the length of it
    # len(input_padded) == n + 4

    input_padded = [input_list[1], input_list[0]]
    input_padded.extend(input_list)
    input_padded.extend([input_list[-1], input_list[-2]])
    
    if odd_padded:
        input_padded.append(input_list[-3])

    n = len(input_padded)
    half = n / 2 - 1
    # print(half)

    high_pass = [0] * half
    low_pass = [0] * half

    for i in range(0, half):
        for j in range(4):
            high_pass[i] += h[j] * input_padded[2 * i + j]
            # if we are doing a compression, the low_pass will be left zeros
            if not compress:
                low_pass[i] += g[j] * input_padded[2 * i + j]

    return high_pass, low_pass, odd_padded

def D4_1D_inv(high_pass, low_pass, odd_padded = False):

    h_inv = [h[2], g[2], h[0], g[0]]
    g_inv = [h[3], g[3], h[1], g[1]]

    n = len(high_pass)

    # len(input_padded) == 2 * n

    input_padded = []
    for i in range(n):
        input_padded.append(high_pass[i])
        input_padded.append(low_pass[i])

    # print(input_padded)

    output_list = [0] * (2 * n - 2)

    for i in range(0, n - 1):
        for j in range(4):
            output_list[2 * i] += h_inv[j] * input_padded[2 * i + j]
            output_list[2 * i + 1] += g_inv[j] * input_padded[2 * i + j]
    
    if odd_padded:
        output_list = output_list[ : -1]

    return output_list


def D4_2D(img, levels = 3):
    
    new_input = img.copy()
    results = []

    for level in range(levels):
        
        res_img, row_odd_padded, col_odd_padded = D4_2D_one_level(new_input, level + 1)
        row, col = res_img.shape
        new_input = res_img[0 : row / 2, 0: col / 2]

        results.append([res_img, row_odd_padded, col_odd_padded])

    return results

def D4_2D_one_level(img, level):

    show_img(img, 'input-level-%d' % level)

    # for level in range(levels):

    row_wave_img = []

    for row in img:

        high_pass, low_pass, row_odd_padded = D4_1D(row)
        row_wave_img.append(high_pass + low_pass)
    
    row_wave_img = np.array(row_wave_img)

    col_row_wave_img = []

    # transpose to iterate over the columns
    for col in row_wave_img.transpose():

        high_pass, low_pass, col_odd_padded = D4_1D(col)
        col_row_wave_img.append(high_pass + low_pass)
    
    # transpose back to its original arrangement
    col_row_wave_img = np.array(col_row_wave_img).transpose()

    show_img(col_row_wave_img, 'output-level-%d' % level)

    return col_row_wave_img, row_odd_padded, col_odd_padded

def D4_2D_inv_one_level(img, row_odd_padded, col_odd_padded, \
                                 level, decompress = False):

    col_inv_img = []
    for col in img.transpose():
        half = len(col) / 2
        high_pass = col[0 : half]
        low_pass = col[half :]
        col_inv = D4_1D_inv(high_pass, low_pass, col_odd_padded)
        col_inv_img.append(col_inv)

    col_inv_img = np.array(col_inv_img).transpose()

    row_col_inv_img = []

    for row in col_inv_img:
        half = len(row) / 2
        high_pass = row[0 : half]
        low_pass = row[half :]
        row_inv = D4_1D_inv(high_pass, low_pass, row_odd_padded)
        row_col_inv_img.append(row_inv)

    row_col_inv_img = np.array(row_col_inv_img)

    if not decompress:
        file_name = 'invesre-level-%d' % level
    else:
        file_name = 'decompress-level-%d' % level
    show_img(row_col_inv_img, file_name)

    return row_col_inv_img

def D4_2D_inv(wavelet_results, levels = 3):

    curr_level = levels - 1
    next_image = wavelet_results[curr_level][0]
    row_odd_padded = wavelet_results[curr_level][1]
    col_odd_padded = wavelet_results[curr_level][2]

    while curr_level >= 0:
        

        level_inv = D4_2D_inv_one_level(next_image, row_odd_padded, \
                            col_odd_padded, level = curr_level + 1)

        if curr_level == 0:
            break

        rows, cols = level_inv.shape

        curr_level -= 1
        next_image = wavelet_results[curr_level][0]
        row_odd_padded = wavelet_results[curr_level][1]
        col_odd_padded = wavelet_results[curr_level][2]

        next_image[ : rows, : cols] = level_inv

def compress_image(img, levels = 2):

    results = D4_2D(img, levels = levels)
    res_img, _, _ = results[-1]
    rows, cols = res_img.shape

    img_wavelet = res_img[ : rows / 2, : cols / 2]
    return img_wavelet

def decompress_image(img_wavelet, levels = 2):

    next_image = img_wavelet.copy()
    while levels > 0:

        rows, cols = next_image.shape
        img_wavelet_expanded = np.zeros([2 * rows, 2 * cols])
        img_wavelet_expanded[: rows, : cols] = next_image

        next_image = D4_2D_inv_one_level(img_wavelet_expanded, \
                                False, False, levels, decompress = True)

        levels -= 1


    return next_image

def test_pywt(l = None, compress = False):

    print('\n\n')
    if l == None:
        l = [3, 7, 1, 1, -2, 5, 4, 6]

    print(l)
    cA, cD = pywt.dwt(l, 'db2')
    print(cA)
    print(cD)

    if compress:
        cD = [0] * len(cD)

    l_inv = pywt.idwt(cA, cD, 'db2')
    print(l_inv)

    w = pywt.Wavelet('db2')
    print(w.dec_hi)
    print(w.dec_lo)

    print(w.rec_hi)
    print(w.rec_lo)


def test_1D(l = None, compress = False):
    if l == None:
        l = [3, 7, 1, 1, -2, 5, 4, 6]

    print(l)
    
    high_pass, low_pass, odd_padded = D4_1D(l, compress)
    print(high_pass)
    print(low_pass)
    
    l_inv = D4_1D_inv(high_pass, low_pass, odd_padded)
    print(l_inv)

def main():

    img = read_image('input.png')

    l = [3, 7, 1, 1, -2, 5, 4, 6]
    
    # test_1D(l)

    # test_pywt(l)

    # only handle the first channel
    img = img[ : , : , 0]
    print(img.shape)

    d4_3_level = D4_2D(img, levels = 3)
    D4_2D_inv(d4_3_level, levels = 3)

    img_wavelet = compress_image(img, levels = 2)
    # print(img_wavelet.shape)
    decompressed_img = decompress_image(img_wavelet, levels = 2)



if __name__ == '__main__':
    main()