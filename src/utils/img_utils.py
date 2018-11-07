from scipy.misc import imsave, imread, imresize
from random import randint
from shutil import copyfile


def random_crop(img, crop_size):
    height, width, _ = img.shape
    start_width = randint(0, width - crop_size)
    start_height = randint(0, height - crop_size)
    return img[start_height: start_height + crop_size, start_width: start_width + crop_size, :]


def central_crop(img, crop_size):
    height, width, _ = img.shape
    start_width = width // 2 - crop_size[0]
    start_height = height // 2 - crop_size[1]
    return img[start_height: start_height + crop_size[0], start_width: start_width + crop_size[1], :]


def copy_images(source_path, destination_path,
                resize=False, resize_factor=1.0):
    if not resize:
        copyfile(source_path, destination_path)
    else:
        img = imread(source_path)
        img = imresize(img, resize_factor)

        imsave(destination_path, img)
