from typing import Any, Callable, Hashable, Mapping, Sequence, Tuple

import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

__all__ = ["SlidingWindowsD"]


class SlidingWindowsD(MapTransform):
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
