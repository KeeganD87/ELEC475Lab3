import torch
from torch.utils.data import Dataset
from torchvision import datasets

def one_hot_encode(target, num_classes=100):
    one_hot_target = torch.zeros(num_classes)
    one_hot_target[target] = 1
    return one_hot_target

class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 100

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        one_hot_target = one_hot_encode(target, self.num_classes)
        return img, one_hot_target