from typing import Any, Callable, Hashable, Mapping, Sequence, Tuple

import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
import numpy as np

__all__ = ["ConvertToMultiChannelBasedOnBratsClassesd", "SlidingWindowsd"]


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        labels: dict = None,
        subregions: list[dict] = None,
    ):
        super().__init__(keys, allow_missing_keys)
        if (labels is None) and (subregions is None):
            labels = {'NCR': 1, 'ED': 2, 'ET': 3}
            subregions = [{'region': 'WT', 'labels': 'NCR+ET+ED'}, 
                    {'region': 'TC', 'labels': 'NCR+ET'}, 
                    {'region': 'ET', 'labels': 'ET'}
            ]
        self.labels = labels
        self.subregions = subregions

    def separate_mask_labels_into_regions(self, mask: NdarrayOrTensor) -> NdarrayOrTensor:
        # maps labels to sub-regions
        # w h d --> c w h d
        subregion_masks = []
        for subregion in self.subregions:
            #subregion_name = subregion["region"]
            subregion_labels = subregion["labels"].split("+")
            subregion_mask = mask == self.labels[subregion_labels[0]]
            if len(subregion_labels) > 1:
                for label in subregion_labels[1:]:
                    subregion_mask += mask == self.labels[label]
            subregion_masks.append(subregion_mask)

        if isinstance(mask, np.ndarray):
            region_mask = np.stack(subregion_masks, axis=0)
        elif isinstance(mask, torch.Tensor):
            region_mask = torch.stack(subregion_masks, dim=0)
        else:
            TypeError('Input needs to be a np.ndarray or torch.Tensor')

        return region_mask

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.separate_mask_labels_into_regions(d[key])
        return d



class SlidingWindowsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        roi_size: Tuple[int, int, int] = [128, 128, 128],
        stride: int = 0.50,
    ):
        super().__init__(keys, allow_missing_keys)
        self.roi_size = roi_size
        self.stride = [int(roi_size[i] * stride) for i in range(3)]

    def sliding_windows_(self, arr):
        c, w, h, d = arr.shape
        assert (
            w % self.roi_size[0] == 0
            and h % self.roi_size[1] == 0
            and d % self.roi_size[2] == 0
        ), "Dimensions should be divisible by roi_size"
        # Calculate the shape of the output windows
        out_w = (w - self.roi_size[0]) // self.stride[0] + 1
        out_h = (h - self.roi_size[1]) // self.stride[1] + 1
        out_d = (d - self.roi_size[2]) // self.stride[2] + 1
        # Unfold the dimensions and reshape
        windows = (
            arr.unfold(1, self.roi_size[0], self.stride[0])
            .unfold(2, self.roi_size[1], self.stride[1])
            .unfold(3, self.roi_size[2], self.stride[2])
        )
        windows = windows.contiguous().view(
            out_w * out_h * out_d,
            c,
            self.roi_size[0],
            self.roi_size[1],
            self.roi_size[2],
        )
        return windows

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.sliding_windows_(d[key])
        return d
