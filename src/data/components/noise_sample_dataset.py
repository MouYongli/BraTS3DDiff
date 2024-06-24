import random

import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset


class NoiseSampleDataset(Dataset):
    def __init__(
        self,
        image_size,
        class_cond=False,
        num_classes=None,
        num_samples=10000,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples
        self.class_cond = class_cond
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = th.randn(*(3, self.image_size, self.image_size))

        out_dict = {}
        if self.class_cond:
            out_dict["y"] = np.array(
                random.randint(0, self.num_classes - 1), dtype=np.int64
            )

        return noise, out_dict
