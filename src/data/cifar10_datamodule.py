import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from .image_datamodule import ImageDataModule


class CIFAR10DataModule(ImageDataModule):
    """`LightningDataModule` for the Cifar10 dataset. 32*32 RGB images of 10 classes.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "/home/sanyal/DATA/cifar10",
        batch_size: int = 128,
        image_size: int = 32,
        class_cond: bool = False,
        n_classes: int = None,
        random_crop: bool = False,
        random_flip: bool = True,
        deterministic: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CIFAR10DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(
            data_dir,
            batch_size,
            image_size,
            class_cond,
            n_classes,
            random_crop,
            random_flip,
            deterministic,
            num_workers,
            pin_memory,
        )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of cifar classes (10).
        """
        return 10

    @property
    def _classes(self) -> Tuple[str, ...]:
        """Get the class names.

        :return: The class names.
        """
        return (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        for split in ["train", "test"]:
            out_dir = os.path.join(self.hparams.data_dir, f"cifar_{split}")
            if os.path.exists(out_dir):
                print(f"skipping split {split} since {out_dir} already exists.")
                continue

            print("downloading...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                dataset = torchvision.datasets.CIFAR10(
                    root=tmp_dir, train=split == "train", download=True
                )

            print("dumping images...")
            os.mkdir(out_dir)
            for i in tqdm(range(len(dataset))):
                image, label = dataset[i]
                filename = os.path.join(out_dir, f"{self._classes[label]}_{i:05d}.png")
                image.save(filename)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        super().setup(stage)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return super().train_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CIFAR10DataModule()
