import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, ceil
from PIL import Image
import math


def read_image(path):
    img = cv2.imread(path)  # cv2.IMREAD_GRAYSCALE)
    size = img.shape
    dimension = (size[0], size[1])

    return img, size, dimension


def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img


def nearest_interpolation(image, dimension):
    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    enlarge_time = int(
        sqrt((dimension[0] * dimension[1]) / (image.shape[0]*image.shape[1])))

    for i in range(dimension[0]):
        for j in range(dimension[1]):
            row = floor(i / enlarge_time)
            column = floor(j / enlarge_time)

            new_image[i, j] = image[row, column]

    return new_image


def bilinear_interpolation(image, dimension):
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    return new_image


def show_result(images_list):
    titles = list(images_list.keys())
    images = list(images_list.values())

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('25 Percent of the original size - PC', fontsize=16)

    axs[0, 0].set_title(titles[0])
    axs[0, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))

    axs[0, 1].set_title(titles[1])
    axs[0, 1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))

    axs[0, 2].set_title(titles[2])
    axs[0, 2].imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))

    axs[1, 0].set_title(titles[3])
    axs[1, 0].imshow(cv2.cvtColor(images[3], cv2.COLOR_BGR2RGB))

    axs[1, 1].set_title(titles[4])
    axs[1, 1].imshow(cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB))

    axs[1, 2].set_title(titles[5])
    axs[1, 2].imshow(cv2.cvtColor(images[5], cv2.COLOR_BGR2RGB))


def main():
    images_list = {}

    # Read Image
    img, size, dimension = read_image("./butterfly.png")
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 25  # percent of original image size
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('My Implementation', fontsize=16)

    # Change image to original size using nearest neighbor interpolation
    nn_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_NEAREST)
    images_list['+ Nearest Neighbor Interpolation'] = nn_img

    nn_img_algo = nearest_interpolation(resized_img, dimension)
    nn_img_algo = Image.fromarray(nn_img_algo.astype('uint8')).convert('RGB')

    near_img = cv2.resize(img, None, fx=0.2, fy=0.2,
                          interpolation=cv2.INTER_NEAREST)
    images_list['- Nearest Neighbor Interpolation'] = near_img

    # Change image to original size using bilinear interpolation
    bil_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    images_list['+ Bilinear Interpolation'] = bil_img

    bil_img_algo = bilinear_interpolation(resized_img, dimension)
    bil_img_algo = Image.fromarray(
        bil_img_algo.astype('uint8')).convert('RGB')

    near_img = cv2.resize(img, None, fx=0.2, fy=0.2,
                          interpolation=cv2.INTER_LINEAR)
    images_list['- Bilinear Interpolation'] = near_img

    # Show PC Result
    show_result(images_list)

    # Show manual results:

    axs[0, 0].set_title("Original")
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    axs[0, 1].set_title("Nearest ZOOM IN by me")
    axs[0, 1].imshow(cv2.cvtColor(np.array(nn_img_algo), cv2.COLOR_BGR2RGB))

    axs[1, 0].set_title("Bilinear ZOOM IN by me")
    axs[1, 0].imshow(cv2.cvtColor(np.array(bil_img_algo), cv2.COLOR_BGR2RGB))

    # plt.grid()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
