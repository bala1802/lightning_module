'''
Author Name: Balaguru Sivasambagupta
Autor Email ID: bala1802@live.com
Github: https://github.com/bala1802
'''

'''
Purpose of this Script:
    - This script will apply data augmentation techniues like `Albumentation` on the dataset
'''

from typing import Any, List, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

'''
This method will apply augmentation and transforms the image for the configured augmentation techniques
In this script , we have applied:
    - `Crop and Pad`
    - `Random Crop
    - `Horizontal Flip`
    - `Cut out`
After applying the augmentation techniques, the augmented data is normalized
'''
def make_transform(image_set: str) -> A.Compose:
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    if image_set == "train":
        return A.Compose(
            [
                A.Sequential(
                    [
                        A.CropAndPad(
                            px=4, keep_size=False
                        ),  # padding of 2, keep_size=True by defaulf
                        A.RandomCrop(32, 32),
                    ]
                ),
                A.HorizontalFlip(),
                # A.ShiftScaleRotate(),
                A.CoarseDropout(1, 16, 16, 1, 16, 16, fill_value=mean, mask_fill_value=None),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )

'''
TODO
'''
class CIFAR10(Dataset):
    def __init__(self, ds: Any, transform: A.Compose):
        self.ds = ds
        self.transforms = transform

    @property
    def classes(self) -> List[str]:
        return self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i) -> Tuple[torch.Tensor, int]:
        image, label = self.ds[i]
        # apply augmentation
        image = self.transforms(image=np.array(image))["image"]
        return image, label