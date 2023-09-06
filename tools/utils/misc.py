'''
Github: https://github.com/bala1802
'''

'''
Purpose of this Script:
    - TODO
'''

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

def per_class_accuracy(model: Any, device: torch.device, data_loader: DataLoader):
    model = model.to(device)
    model.eval()

    classes = data_loader.dataset.classes
    nc = len(classes)
    class_correct = list(0.0 for i in range(nc))
    class_total = list(0.0 for i in range(nc))
    with torch.inference_mode():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(nc):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("[x] Accuracy of ::")
    for i in range(nc):
        print("\t[*] %8s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))
