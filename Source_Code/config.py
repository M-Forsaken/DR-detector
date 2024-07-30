import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

WORKINGDIR = os.getcwd()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
NUM_WORKERS = 6
CHECKPOINT_FILE_TRAIN = "EfficientNetB3.pth.tar"
CHECKPOINT_FILE_TRAIN_BLEND = "Linear.pth.tar"
SAVE_PATH = WORKINGDIR + "/Models/"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True
RANDAUGMENT_MODE = False


# tqdm bar format
BARFORMAT = "{desc}{percentage:3.0f}%│{bar:30}│total: {n_fmt}/{total_fmt} [{elapsed} - {remaining},{rate_fmt}{postfix}]"
DELAYTIME = 0.1
COLOR = "red"
COLOR_COMPLETE = "green"
ASCII = ' ▌█'
# data augmentation for images:


def randAugment(N=2, M=4, p=1.0, mode="all", cut_out=False):
    """
    Examples:
        >>> # M from 0 to 20
        >>> transforms = randAugment(N=3, M=8, p=0.8, mode='all', cut_out=False)
    """
    # Magnitude(M) search space
    scale = np.linspace(0, 0.4, 20)
    translate = np.linspace(0, 0.4, 20)
    rot = np.linspace(0, 30, 20)
    shear_x = np.linspace(0, 20, 20)
    shear_y = np.linspace(0, 20, 20)
    contrast = np.linspace(0.0, 0.4, 20)
    bright = np.linspace(0.0, 0.4, 20)
    sat = np.linspace(0.0, 0.2, 20)
    hue = np.linspace(0.0, 0.2, 20)
    shar = np.linspace(0.0, 0.9, 20)
    blur = np.linspace(0, 0.2, 20)
    noise = np.linspace(0, 1, 20)
    cut = np.linspace(0, 0.6, 20)
    # Transformation search space
    Aug = [  # geometrical
        A.Affine(scale=(1.0 - scale[M], 1.0 + scale[M]), p=p),
        A.Affine(translate_percent=(-translate[M], translate[M]), p=p),
        A.Affine(rotate=(-rot[M], rot[M]), p=p),
        A.Affine(shear={'x': (-shear_x[M], shear_x[M])}, p=p),
        A.Affine(shear={'y': (-shear_y[M], shear_y[M])}, p=p),
        # Color Based
        A.RandomBrightnessContrast(
            brightness_limit=bright[M], contrast_limit=contrast[M], p=p),
        A.ColorJitter(brightness=0.0, contrast=0.0,
                      saturation=sat[M], hue=0.0, p=p),
        A.ColorJitter(brightness=0.0, contrast=0.0,
                      saturation=0.0, hue=hue[M], p=p),
        A.Sharpen(alpha=(0.1, shar[M]), lightness=(0.5, 1.0), p=p),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=1),
            A.Blur(blur_limit=3, p=1), ]),
        A.GaussNoise(var_limit=(
            8.0 * noise[M], 64.0 * noise[M]), per_channel=True, p=p)
    ]
    # Sampling from the Transformation search space
    if mode == "geo":
        transforms = A.SomeOf(Aug[0:5], N)
    elif mode == "color":
        transforms = A.SomeOf(Aug[5:], N)
    else:
        transforms = A.SomeOf(Aug, N)

    if cut_out:
        cut_trans = A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=16,
                            max_width=16, fill_value=0, p=1),
            A.GridDropout(ratio=cut[M], p=1),
        ], p=cut[M])
        transforms = A.Compose([transforms, cut_trans])

    return transforms


def Transforms(Randaugmentmode=False, Size=150, Crop_Size=120):
    if (Randaugmentmode == True):
        # randaugment transform (only use this setting on low batches or if you have a powerful CPU)
        size_transforms = A.Compose(
            [
                A.Resize(width=Size, height=Size),
                A.RandomCrop(height=Crop_Size, width=Crop_Size),
                A.Normalize(
                    mean=[0.3199, 0.2240, 0.1609],
                    std=[0.3020, 0.2183, 0.1741],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
        randAug_transforms = randAugment(
            N=3, M=8, p=0.8, mode='all', cut_out=True)

        train_transforms = A.Compose([randAug_transforms, size_transforms])

        val_transforms = A.Compose(
            [
                A.Resize(height=Crop_Size, width=Crop_Size),
                A.Normalize(
                    mean=[0.3199, 0.2240, 0.1609],
                    std=[0.3020, 0.2183, 0.1741],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        train_transforms = A.Compose(
            [
                A.Resize(width=Size, height=Size),
                A.RandomCrop(height=Crop_Size, width=Crop_Size),
                A.Normalize(
                    mean=[0.3199, 0.2240, 0.1609],
                    std=[0.3020, 0.2183, 0.1741],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        val_transforms = A.Compose(
            [
                A.Resize(height=Crop_Size, width=Crop_Size),
                A.Normalize(
                    mean=[0.3199, 0.2240, 0.1609],
                    std=[0.3020, 0.2183, 0.1741],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    return train_transforms, val_transforms


def Get_Settings(SIZE=150, CROP_SIZE=120):
    if SIZE == 150 and CROP_SIZE == 120:
        BATCH_SIZE_TRAIN = 64
        BATCH_SIZE_CHECKACC = 256
        return SIZE, CROP_SIZE, BATCH_SIZE_TRAIN, BATCH_SIZE_CHECKACC
    if SIZE == 330 and CROP_SIZE == 300:
        BATCH_SIZE_TRAIN = 16
        BATCH_SIZE_CHECKACC = 64
        return SIZE, CROP_SIZE, BATCH_SIZE_TRAIN, BATCH_SIZE_CHECKACC
    if SIZE == 550 and CROP_SIZE == 500:
        BATCH_SIZE_TRAIN = 4
        BATCH_SIZE_CHECKACC = 16
        return SIZE, CROP_SIZE, BATCH_SIZE_TRAIN, BATCH_SIZE_CHECKACC
    if SIZE == 800 and CROP_SIZE == 728:
        BATCH_SIZE_TRAIN = 2
        BATCH_SIZE_CHECKACC = 8
        return SIZE, CROP_SIZE, BATCH_SIZE_TRAIN, BATCH_SIZE_CHECKACC

# initiate


# Settings includes:
# (800,728)|(550,500)|(330,300)|(150,120)
SIZE, CROP_SIZE, BATCH_SIZE_TRAIN, BATCH_SIZE_CHECKACC = Get_Settings(
    SIZE=800, CROP_SIZE=728)

train_transforms, val_transforms = Transforms(
    Randaugmentmode=True, Size=SIZE, Crop_Size=CROP_SIZE)
