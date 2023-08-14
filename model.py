import torch
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

import config

class LitResnet(LightningModule):
    
    def __init__(self, learning_rate=config.LEARNING_RATE, drop=config.DROP_VALUE, 
                 norm=config.BATCH_NORMALIZATION, groupsize=config.GROUP_SIZE):
        
        super().__init__()

        self.save_hyperparameters()

        self.num_classes =10
        self.lr = learning_rate
        self.drop = drop
        self.normalization=norm
        self.groupsize = groupsize

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.user_norm(self.normalization,64,self.groupsize),
            nn.Dropout(self.drop),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            self.user_norm(self.normalization,128,self.groupsize),
            nn.Dropout(self.drop))
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.user_norm(self.normalization,128,self.groupsize),
            nn.Dropout(self.drop),
            nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.user_norm(self.normalization,128,self.groupsize),
            nn.Dropout(self.drop)
             )

        # CONVOLUTION BLOCK 2
      	# Layer 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.user_norm(self.normalization,256,self.groupsize),
            nn.Dropout(self.drop),
	        nn.MaxPool2d(2,2),
            nn.ReLU(),
            self.user_norm(self.normalization,256,self.groupsize),
            nn.Dropout(self.drop)

             )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
	        nn.MaxPool2d(2,2),
            nn.ReLU(),
            self.user_norm(self.normalization,512,self.groupsize),
            nn.Dropout(self.drop)
            )
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.user_norm(self.normalization,512,self.groupsize),
            nn.Dropout(self.drop),
            nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.user_norm(self.normalization,512,self.groupsize),
            nn.Dropout(self.drop))

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False))
        self.model = nn.Sequential(self.convblock1 ,self.convblock2, self.convblock3, self.convblock4)

    def user_norm(self, norm, channels, groupsize=1):

        if norm == config.BATCH_NORMALIZATION:
            return nn.BatchNorm2d(channels)
        elif norm == config.LAYER_NORMALIZATION:
            return nn.GroupNorm(1, channels)
        elif norm == config.GROUP_NORMALIZATION:
            return nn.GroupNorm(groupsize, channels)
    
    def forward(self, x):

        x = self.convblock1(x)
        x = x + self.res1 (x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x + self.res2 (x)
        x = self.convblock4(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task =  'multiclass', num_classes = self.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=config.WEIGHT_DECAY,
        )

        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=config.STEPS_PER_EPOCH
            ),
            "interval": "step"
        }

        return {"optimizer":optimizer, "lr_scheduler": scheduler_dict}
    

