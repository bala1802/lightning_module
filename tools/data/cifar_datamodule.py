'''
Author Name: Balaguru Sivasambagupta
Autor Email ID: bala1802@live.com
Github: https://github.com/bala1802
'''

'''
Purpose of this Script:
    - A LightningDataModule is initialized to work on the CIFAR10 dataset
'''

from typing import Optional, Union

import albumentations as A
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .components.cifar10 import CIFAR10, make_transform


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 512, num_workers: int = 0, 
                 pin_memory: bool = False, train_augments: Union[A.Compose, None] = None):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = (
            make_transform("train") if train_augments is None else train_augments
        )
        self.val_transforms = make_transform("val")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    '''
        - The dataset is downloaded and stored inside the specified directory
        - Both the `train` and `test/val` datasets are downloaded
    '''
    def prepare_data(self):
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    '''
     A simple validation is done to check whether the `train` and `test` data are already instantiated or not
    '''
    def setup(self, stage=None) -> None:
        if not self.data_train and not self.data_val:
            self.data_train = datasets.CIFAR10(self.hparams.data_dir, train=True)
            self.data_val = datasets.CIFAR10(self.hparams.data_dir, train=False)

    '''
    `train` data loader instantiation
    '''
    def train_dataloader(self):
        return DataLoader(
            dataset=CIFAR10(self.data_train, self.train_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    '''
    `validation` data loader instantiation
    '''
    def val_dataloader(self):
        return DataLoader(
            dataset=CIFAR10(self.data_val, self.val_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )