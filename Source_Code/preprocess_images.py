import os
import sys
from time import sleep
import numpy as np
from PIL import Image, ImageFile
from multiprocessing import Pool
from tqdm import tqdm
import cv2
import config
from utils import Slowprint

ImageFile.LOAD_TRUNCATED_IMAGES = True


def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percetage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """
    percentage = 0.02

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row: max_row + 1, min_col: max_col + 1]
    return Image.fromarray(im_crop)


def resize_maintain_aspect(image, desired_size):
    """
    Got this from some stackoverflow ,
    this will add padding to maintain the aspect ratio.
    """
    old_size = image.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def save_single(args):
    img_file, input_path_folder, output_path_folder, output_size = args
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_original)
    image = resize_maintain_aspect(image, desired_size=output_size[0])
    image.save(os.path.join(output_path_folder + img_file))


def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    if not output_size:
        while True:
            Slowprint(
                "Output Filesize Will Be Set To Default Which Is 1000x1000.")
            Slowprint("Continue? Y/N")
            I = input()
            if (I == "n" or I == "N"):
                sys.exit()
            else:
                if (I == "y" or I == "Y"):
                    os.system("cls")
                    break
            os.system("cls")
        output_size = (1000, 1000)

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]
    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs), leave=False, delay=config.DELAYTIME,
             colour=config.COLOR, desc="Preprocessing Images: ", bar_format=config.BARFORMAT))
    Slowprint("Process Completed.")
