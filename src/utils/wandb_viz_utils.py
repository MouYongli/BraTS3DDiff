from monai.config.type_definitions import NdarrayOrTensor
from tqdm import tqdm
import wandb

CLASS_NAMES = {'WT':'Whole-Tumor', 'TC': 'Tumor-Core', 'ET': 'Enhancing-Tumor'}
CLASS_NAMES_ = {'WT':'Whole Tumor', 'TC': 'Tumor Core', 'ET': 'Enhancing Tumor'}

def log_data_samples_into_tables(
    sample_image: NdarrayOrTensor,
    sample_label: NdarrayOrTensor,
    subregions: list=None,
    split: str = None,
    data_idx: int = None,
    table: wandb.Table = None,
):
    assert len(sample_image.shape) == len(sample_image.shape) == 4
    num_channels, _, _, num_slices = sample_image.shape
    for slice_idx in range(num_slices):
        ground_truth_wandb_images = []
        masks = {}
        for i in range(len(subregions)):
            region_name = CLASS_NAMES[subregions[i]]
            region_name_ = CLASS_NAMES_[subregions[i]]
            masks[f'ground-truth/{region_name}'] = {
                "mask_data": sample_label[i, :, :, slice_idx] * (i+1),
                "class_labels": {0: "background", i+1: region_name_},
            }
        for channel_idx in range(num_channels):
            img = sample_image[channel_idx, :, :, slice_idx]
            ground_truth_wandb_images.append(wandb.Image(img, masks=masks))
        table.add_data(split, data_idx, slice_idx, *ground_truth_wandb_images)
    return table


'''
def log_data_samples_into_tables(
    sample_image: NdarrayOrTensor,
    sample_label: NdarrayOrTensor,
    subregions: list=None,
    split: str = None,
    data_idx: int = None,
    table: wandb.Table = None,
):
    num_channels, _, _, num_slices = sample_image.shape
    with tqdm(total=num_slices, leave=False) as progress_bar:
        for slice_idx in range(num_slices):
            ground_truth_wandb_images = []
            for channel_idx in range(num_channels):
                ground_truth_wandb_images.append(
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Whole-Tumor": {
                                "mask_data": sample_label[0, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Whole Tumor"},
                            },
                            "ground-truth/Tumor-Core": {
                                "mask_data": sample_label[1, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Tumor Core"},
                            },
                            "ground-truth/Enhancing-Tumor": {
                                "mask_data": sample_label[2, :, :, slice_idx] * 3,
                                "class_labels": {0: "background", 3: "Enhancing Tumor"},
                            },
                        },
                    )
                )
            table.add_data(split, data_idx, slice_idx, *ground_truth_wandb_images)
            progress_bar.update(1)
    return table
'''