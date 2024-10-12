import os
import shutil
from glob import glob
from typing import Any, List, Optional, Tuple

import lightning.pytorch as pl
import nibabel as nib
import numpy as np
import torch
import yaml
from einops import rearrange, reduce, repeat
from monai.transforms import (
    Compose,
    CropForegroundd,
    DivisiblePadd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianSharpend,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ToTensord,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate

from src.datasets.transforms.transforms import SlidingWindowsD
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def custom_collate(batch):
    datas = []
    file_ids = []
    for item in batch:
        data, file_id = item
        datas.append(data)
        file_ids.append(file_id)

    datas = default_collate(datas)
    return datas, file_ids


class BraTSDataset(Dataset):
    def __init__(
        self,
        image_path: List[str],
        transforms: Optional[Compose] = None,
        data_dir: str = "./data/BraTS",
        mode: str = "train",
        preprocess_mask_labels: bool = False,
        dim_order: str = "w h d",
        labels: dict = None,
        subregions: dict = None,
        im_channels: list = None,
        patch_sizes: list = None,
        sep: str = None,
        ext: str = None,
        thresh: int = 0,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.mode = mode
        self.predict = mode == "predict"
        if not os.path.exists(self.data_dir):
            log.warning("BraTS dataset does not exist")
            raise FileNotFoundError
        self.image_path = image_path
        self.transforms = transforms
        self.preprocess_mask_labels = preprocess_mask_labels
        self.dim_order = dim_order
        self.labels = labels
        self.subregions = subregions
        self.im_channels = im_channels
        self.patch_sizes = patch_sizes
        self.sep = sep
        self.ext = ext
        self.thresh = thresh

    def read_data(self, data_path):
        file_id = os.path.split(data_path)[1]
        im_channel_paths = [
            os.path.join(data_path, f"{file_id}{self.sep}{channel}{self.ext}")
            for channel in self.im_channels
        ]
        image = np.stack(
            [nib.load(p).get_fdata().astype(np.float32) for p in im_channel_paths],
            axis=0,
        )
        image = rearrange(image, f"c {self.dim_order} -> c w h d")

        if self.mode in ["train", "val"]:
            seg_path = os.path.join(data_path, f"{file_id}{self.sep}seg{self.ext}")
            mask = nib.load(seg_path).get_fdata().astype(np.float32)
            mask = rearrange(mask, f"{self.dim_order} -> w h d")
            return {"image": image, "mask": mask}

        elif self.mode == "test":
            seg_path = os.path.join(data_path, f"{file_id}{self.sep}seg{self.ext}")
            mask = nib.load(seg_path).get_fdata().astype(np.float32)
            mask = rearrange(mask, f"{self.dim_order} -> w h d")
            return ({"image": image, "mask": mask}, file_id)

        elif self.mode == "predict":
            return (
                {
                    "image": image,
                },
                file_id,
            )

        else:
            raise ValueError("mode must be in ['train', 'val', 'test', 'predict']")

    def separate_mask_labels_into_regions(self, mask: np.ndarray) -> np.ndarray:
        # maps labels to sub-regions
        # w h d --> c w h d
        subregion_masks = []
        for subregion in self.subregions:
            subregion_name = subregion["region"]
            subregion_labels = subregion["labels"].split("+")
            subregion_mask = mask == self.labels[subregion_labels[0]]
            if len(subregion_labels) > 1:
                for label in subregion_labels[1:]:
                    subregion_mask += mask == self.labels[label]
            subregion_masks.append(subregion_mask > 0)
        return np.stack(subregion_masks, axis=0)

    def label_patches(self, mask):
        # patchify masks into series of (N*N*N) patches
        # and compute fraction of foreground pixels of a mask channel in every patch
        # input: (B)xCxWxHxD
        # output: (B)x1x(W//N)x(H//N)x(D//N)
        # label patches as tumor/non-tumor based on WT tumor vol frac
        # mask[0]: only consider WT
        if len(mask.shape) == 4:
            c, w, h, d = mask.shape
            mask = mask[0].unsqueeze(0)
        elif len(mask.shape) == 5:
            b, c, w, h, d = mask.shape
            mask = mask[:, 0].unsqueeze(1)

        patch_tumor_vols = {}
        for patch_size in self.patch_sizes:
            assert (
                w % patch_size == 0 and h % patch_size == 0 and d % patch_size == 0
            ), f"w, h, and d must be divisible by patch_size={patch_size}"
            if len(mask.shape) == 4:
                mask_patch = mask.reshape(
                    1,
                    w // patch_size,
                    patch_size,
                    h // patch_size,
                    patch_size,
                    d // patch_size,
                    patch_size,
                )
                patch_tumor_vol = mask_patch.sum(axis=(2, 4, 6)) / (
                    patch_size * patch_size * patch_size
                )
            elif len(mask.shape) == 5:
                mask_patch = mask.reshape(
                    b,
                    1,
                    w // patch_size,
                    patch_size,
                    h // patch_size,
                    patch_size,
                    d // patch_size,
                    patch_size,
                )
                patch_tumor_vol = mask_patch.sum(axis=(3, 5, 7)) / (
                    patch_size * patch_size * patch_size
                )

            # label patches as tumor(1)/non-tumor(0) based on patch tumor vol frac
            patch_tumor_vol[patch_tumor_vol > self.thresh] = 1.0
            patch_tumor_vol[patch_tumor_vol <= self.thresh] = 0.0
            patch_tumor_vols[patch_size] = patch_tumor_vol

        return patch_tumor_vols

    def zero_pad(self, data):
        # zero pad to make the img and mask size (256 x 256 x 256)
        pad_width = [(0, 0), (8, 8), (8, 8), (50, 51)]  # Padding for (w, h, d)
        data["image"] = np.pad(
            data["image"], pad_width, mode="constant", constant_values=0
        )
        data["mask"] = np.pad(
            data["mask"], pad_width, mode="constant", constant_values=0
        )
        return data

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, index: int) -> Any:
        """Mask with shape C x W x H x D image with shape C x W x H x D volume_map with shape C x
        W//N x H//N x D//N."""
        if self.mode in ["train", "val"]:
            data = self.read_data(self.image_path[index])
            data["mask"] = self.separate_mask_labels_into_regions(data["mask"]).astype(
                np.uint8
            )
            image = data["image"]
            foreground = reduce(image, "c w h d -> () w h d", "sum")
            foreground = np.where(foreground > 0, 1, 0).astype(np.float32)
            data["foreground"] = foreground
            data = self.transforms(data)
            data["patch_tumor_labels"] = self.label_patches(data["mask"])
            return data

        elif self.mode == "test":
            data, file_id = self.read_data(self.image_path[index])
            data["mask"] = self.separate_mask_labels_into_regions(data["mask"]).astype(
                np.uint8
            )
            image = data["image"]
            foreground = reduce(image, "c w h d -> () w h d", "sum")
            foreground = np.where(foreground > 0, 1, 0).astype(np.float32)
            data["foreground"] = foreground
            data = self.transforms(data)
            return (data, file_id)

        elif self.mode == "predict":
            data, file_id = self.read_data(self.image_path[index])
            image = data["image"]
            foreground = reduce(image, "c w h d -> () w h d", "sum")
            foreground = np.where(foreground > 0, 1, 0).astype(np.float32)
            data["foreground"] = foreground
            data = self.transforms(data)
            return (data, file_id)

        else:
            raise ValueError("mode must be in ['train', 'val', 'test', 'predict']")


class BraTSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/BraTS",
        val_split: float = 0.20,
        roi_size: Tuple[int, int, int] = [128, 128, 128],
        stride: float = 0.50,
        batch_size: int = 8,
        seed: int = 42,
        num_workers: int = 1,
        predict_set: str = "test",
        dim_order: str = "w h d",
        num_targets: int = 3,
        num_modalities: int = 4,
        labels: dict = None,
        subregions: dict = None,
        im_channels: list = None,
        patch_sizes: list = [16, 32],
        sep: str = None,
        ext: str = None,
        thresh: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        np.random.seed(self.hparams.seed)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        w, h, d = self.hparams.roi_size
        self.num_workers = num_workers
        self.subregions_names = [subregion["region"] for subregion in subregions]
        self.im_channels = im_channels
        self.train_transforms = Compose(
            [
                RandSpatialCropd(
                    keys=["image", "mask", "foreground"],
                    roi_size=[w, h, d],
                    random_size=False,
                    allow_missing_keys=True,
                ),
                RandFlipd(
                    keys=["image", "mask", "foreground"],
                    prob=0.5,
                    spatial_axis=-1,
                    allow_missing_keys=True,
                ),
                RandFlipd(
                    keys=["image", "mask", "foreground"],
                    prob=0.5,
                    spatial_axis=-2,
                    allow_missing_keys=True,
                ),
                RandFlipd(
                    keys=["image", "mask", "foreground"],
                    prob=0.5,
                    spatial_axis=-3,
                    allow_missing_keys=True,
                ),
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=True,
                    channel_wise=True,
                    allow_missing_keys=True,
                ),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandAdjustContrastd(
                    keys=["image"],
                    prob=0.15,
                    gamma=(0.65, 1.5),
                    allow_missing_keys=True,
                ),
                ToTensord(keys=["image", "mask", "foreground"]),
            ]
        )

        self.val_transforms = Compose(
            [
                RandSpatialCropd(
                    keys=["image", "mask", "foreground"],
                    roi_size=[w, h, d],
                    random_size=False,
                    allow_missing_keys=True,
                ),
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=True,
                    channel_wise=True,
                    allow_missing_keys=True,
                ),
                ToTensord(keys=["image", "mask", "foreground"]),
            ]
        )

        self.test_transforms = Compose(
            [  # DivisiblePadd(keys=['image','mask'], k=[w, h, d], allow_missing_keys=True),
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=True,
                    channel_wise=True,
                    allow_missing_keys=True,
                ),
                ToTensord(keys=["image", "mask", "foreground"]),
            ]
        )

        self.predict_transforms = Compose(
            [
                NormalizeIntensityd(
                    keys=["image"],
                    nonzero=True,
                    channel_wise=True,
                    allow_missing_keys=True,
                ),
                ToTensord(keys=["image", "foreground"]),
            ]
        )

        self.setup()

    def _load_image_paths(self, split="train"):
        split_dir = os.path.join(self.hparams.data_dir, split)
        if os.path.exists(split_dir):
            log.info(f"loading {split} dataset")
            im_names = sorted(os.listdir(split_dir))
            if len(im_names) > 0:
                return [os.path.join(split_dir, f) for f in im_names]
            else:
                raise ValueError(f"{split} dir is empty")
        else:
            raise FileNotFoundError(f"{split} dir cannot be found")

    def _create_val_subset(self):
        # creates val data by splitting train data
        # split train data into train and val sets
        # val set should not contain any case_id in train set
        # BraTS-GLI-case_id-time_id
        train_dir = os.path.join(self.hparams.data_dir, "train")
        val_dir = os.path.join(self.hparams.data_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        case_ids = sorted(
            list({fname.split("-")[2] for fname in os.listdir(train_dir)})
        )
        train_case_ids, val_case_ids = train_test_split(
            case_ids, test_size=self.hparams.val_split, random_state=self.hparams.seed
        )
        val_paths = []
        train_fnames = sorted(os.listdir(train_dir))
        for fname in train_fnames:
            case_id = fname.split("-")[2]
            if case_id in val_case_ids:
                val_paths.append(os.path.join(val_dir, fname))
                shutil.move(os.path.join(train_dir, fname), val_dir)
        return val_paths

    def setup(self, stage: Optional[str] = None):
        try:
            val_paths = self._load_image_paths("val")
        except:
            log.warning("Validation data not found, Creating validation set")
            val_paths = self._create_val_subset()

        train_paths = self._load_image_paths("train")

        test_paths = self._load_image_paths("test")

        self.data_train = BraTSDataset(
            image_path=train_paths,
            transforms=self.train_transforms,
            data_dir=self.hparams.data_dir,
            dim_order=self.hparams.dim_order,
            labels=self.hparams.labels,
            subregions=self.hparams.subregions,
            im_channels=self.hparams.im_channels,
            patch_sizes=self.hparams.patch_sizes,
            sep=self.hparams.sep,
            ext=self.hparams.ext,
            mode="train",
            thresh=self.hparams.thresh,
        )

        self.data_val = BraTSDataset(
            image_path=val_paths,
            transforms=self.val_transforms,
            data_dir=self.hparams.data_dir,
            dim_order=self.hparams.dim_order,
            labels=self.hparams.labels,
            subregions=self.hparams.subregions,
            im_channels=self.hparams.im_channels,
            patch_sizes=self.hparams.patch_sizes,
            sep=self.hparams.sep,
            ext=self.hparams.ext,
            mode="val",
            thresh=self.hparams.thresh,
        )

        self.data_test = BraTSDataset(
            image_path=val_paths,
            transforms=self.test_transforms,
            data_dir=self.hparams.data_dir,
            dim_order=self.hparams.dim_order,
            labels=self.hparams.labels,
            subregions=self.hparams.subregions,
            im_channels=self.hparams.im_channels,
            patch_sizes=self.hparams.patch_sizes,
            sep=self.hparams.sep,
            ext=self.hparams.ext,
            mode="test",
            thresh=self.hparams.thresh,
        )

        self.data_predict = BraTSDataset(
            image_path=test_paths,
            transforms=self.predict_transforms,
            data_dir=self.hparams.data_dir,
            dim_order=self.hparams.dim_order,
            labels=self.hparams.labels,
            subregions=self.hparams.subregions,
            im_channels=self.hparams.im_channels,
            patch_sizes=self.hparams.patch_sizes,
            sep=self.hparams.sep,
            ext=self.hparams.ext,
            mode="predict",
            thresh=self.hparams.thresh,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    with open("/home/sanyal/Projects/BraTS3DDiff/configs/data/brats23.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("_target_")
    cfg["data_dir"] = "/home/sanyal/Projects/BraTS3DDiff/data/BraTS-Data/BraTS2024-GLI"
    cfg["patch_sizes"] = [2, 4, 8, 16, 32]
    a = BraTSDataModule(**cfg)

    train_data = a.train_dataloader()
    data = iter(train_data).__next__()
    image = data["image"]
    print("train loader data", image.shape)
    mask = data["mask"]
    print("train loader mask", mask.shape)
    patch_vols = data["volume_maps"]
    print(
        "train loader patch vols", [patch_vols[i].shape for i in range(len(patch_vols))]
    )
