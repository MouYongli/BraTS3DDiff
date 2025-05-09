import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm
from src.utils.wandb_viz_utils import log_data_samples_into_tables

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LogDataSamplesCallback(Callback):
    def __init__(self, max_train_images_visualized=10, max_val_images_visualized=10):
        super().__init__()
        self.table = wandb.Table(
            columns=[
                "Split",
                "Data Index",
                "Slice Index",
                "Image-Channel-0",
                "Image-Channel-1",
                "Image-Channel-2",
                "Image-Channel-3",
            ]
        )
        self.max_train_images_visualized = max_train_images_visualized
        self.max_val_images_visualized = max_val_images_visualized

    def on_train_start(self, trainer, pl_module):
        subregions = trainer.datamodule.subregions_names
        # Generate visualizations for train_dataset
        train_dataset = trainer.datamodule.data_train
        max_samples = (
            min(self.max_train_images_visualized, len(train_dataset))
            if self.max_train_images_visualized > 0
            else len(train_dataset)
        )

        log.info('Generating Train Dataset Visualizations:')
        for data_idx in range(max_samples):
            sample = train_dataset[data_idx]
            sample_image = sample["image"].detach().cpu()
            sample_label = sample["mask"].detach().cpu()
            self.table = log_data_samples_into_tables(
                sample_image,
                sample_label,
                subregions,
                split="train",
                data_idx=data_idx,
                table=self.table,
            )

        # Generate visualizations for val_dataset
        val_dataset = trainer.datamodule.data_val
        max_samples = (
            min(self.max_val_images_visualized, len(val_dataset))
            if self.max_val_images_visualized > 0
            else len(val_dataset)
        )

        log.info('Generating Validation Dataset Visualizations:')
        for data_idx in range(max_samples):
            sample = val_dataset[data_idx]
            sample_image = sample["image"].detach().cpu()
            sample_label = sample["mask"].detach().cpu()
            self.table = log_data_samples_into_tables(
                sample_image,
                sample_label,
                subregions,
                split="val",
                data_idx=data_idx,
                table=self.table,
            )

        # Log the table to your dashboard
        wandb.log({"Tumor-Segmentation-Data": self.table})
