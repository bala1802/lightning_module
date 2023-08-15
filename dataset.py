import torch
import torch.nn.functional as F
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

import config
from dataTransformer import DATA_AUGMENTATION, construct_transform_object

class CIFAR10DataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir=config.DATA_DIR, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        self.train_dataset = DATA_AUGMENTATION(data_dir=self.data_dir, train=True, 
                          transform=construct_transform_object(train=True))
        self.test_dataset = DATA_AUGMENTATION(data_dir=self.data_dir, train=False, 
                          transform=construct_transform_object(train=False))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



