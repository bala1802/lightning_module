# Lightning Module for ERA-Session-12 Assignment

This repository is written to handle the PyTorch Lightning libraries for the Session-12 Assignment. The Code block is divided into three parts:

- `Data` module
- `Models` module
- `Utils`

#### Data Module

- The CIFAR10 dataset is loaded and packaged into `LightningDataModule`
- The data augmentation is applied to the `Train` dataset
- To apply augmentation, a helper component is created to apply `Albumentation` techniques
- Constructed the `data loader` specific to the `Train`, `Test` and `Validation`

#### Models Module

- A ResNet architecture is initialized
- The ResNet model is packaged into a `LightningModule`
- In the `LightningModule`, the model is initialized, and trained. Much more details explained in `Utils Module`

#### Utils

The `utils` holds all the helper scripts.

- `gradcam.py` - This holds the `gradcam` technique applied on the images for the specified `layer`
- `helper.py` - Using Auto Find Learning Rate Scheduler, the `learning rate` is calculated. Post that, the model is trained
- `misc.py` - The Accuracy for each class is calculated
- `plotting.py` - To plot the misclassified images