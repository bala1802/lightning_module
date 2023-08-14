'''
The purpose of this script is to apply albumentation (data augmentation) 
technique on the train and test dataset
'''

from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

class DATA_AUGMENTATION(datasets.CIFAR10):

    def __init__(self, data_dir, train=True, download=False, transform=None):
        super().__init__(root=data_dir, train=train, download=download, transform=transform)
    
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    
def construct_transform_object(train=True):
    if train:
        return A.Compose(
            [
                A.PadIfNeeded(min_height=config.AUGMENTATION_PADDING_MIN_HEIGHT, 
                              min_width=config.AUGMENTATION_PADDING_MIN_WIDTH),
                A.RandomCrop(width=config.AUGMENTATION_RANDOM_CROP_WIDTH, 
                             height=config.AUGMENTATION_RANDOM_CROP_WIDTH, 
                             always_apply=config.AUGMENTATION_RANDOM_CROP_ALWAYS_APPLY),
                A.HorizontalFlip(p=config.AUGMENTATION_HORIZONTAL_FLIP_PROB),
                A.CoarseDropout(max_holes=config.AUGMENTATION_CUTOUT_MAX_HOLES,
                                min_holes=config.AUGMENTATION_CUTOUT_MAX_HOLES,
                                max_height=config.AUGMENTATION_CUTOUT_MAX_HEIGHT,
                                max_width=config.AUGMENTATION_CUTOUT_MAX_WIDTH,
                                p=config.AUGMENTATION_CUTOUT_PROB,
                                fill_value=tuple([x * 255.0 for x in config.CIFAR_10_DATASET_MEAN]),
                                min_height=config.AUGMENTATION_CUTOUT_MIN_HEIGHT,
                                min_width=config.AUGMENTATION_CUTOUT_MIN_WIDTH,
                                mask_fill_value=config.AUGMENTATION_MASK_FILL_VALUE),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=config.CIFAR_10_DATASET_MEAN,
                    std=config.CIFAR_10_DATASET_STANDARD_DEVIATION,
                    p=1.0,
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )