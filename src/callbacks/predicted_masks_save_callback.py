from lightning.pytorch.callbacks import Callback
import os
import nibabel
import torch
from einops import repeat, rearrange, reduce

class PredictedMasksSaveCallBack(Callback):
    def __init__(self, save_dir):
        self.save_dir=save_dir

    @staticmethod
    def mask_regions_to_labels(data_dir, mask_regions, labels, dim_order):
        # convert multichannel mask regions to single channel predicted mask label
        if os.path.split(data_dir)[1] == 'BraTS2023-GLI':
            p_wt, p_tc, p_et = mask_regions[:, 0].unsqueeze(1), mask_regions[:, 1].unsqueeze(1), mask_regions[:, 2].unsqueeze(1)
            mask_labels = torch.zeros_like(p_wt, dtype=torch.uint8)
            mask_labels[p_wt==1] = labels['ED']    #Class 2: ED
            mask_labels[p_tc==1] = labels['NCR']    #Class 1: NCR
            mask_labels[p_et==1] = labels['ET']   #Class 3: ET

        elif os.path.split(data_dir)[1] == 'BraTS2024-GLI':
            p_et, p_netc, p_snfh, p_rc = mask_regions[:, 2].unsqueeze(1), mask_regions[:, 3].unsqueeze(1), mask_regions[:, 4].unsqueeze(1), mask_regions[:, 5].unsqueeze(1)
            mask_labels = torch.zeros_like(p_et, dtype=torch.uint8)
            mask_labels[p_netc==1] = labels['NETC']
            mask_labels[p_snfh==1] = labels['SNFH']
            mask_labels[p_et==1] = labels['ET']
            mask_labels[p_rc==1] = labels['RC']

        mask_labels = mask_labels.squeeze(1)
        mask_labels = rearrange(mask_labels, f"b w h d -> b {dim_order}")
        return mask_labels
    
    @staticmethod
    def load_affine_and_header(data_dir, file_id, im_channels, sep, ext, split='test'):
        im_path = os.path.join(data_dir,split,file_id,f"{file_id}{sep}{im_channels[0]}{ext}")
        ni_data = nibabel.load(im_path)
        affine, header = ni_data.affine, ni_data.header
        return affine, header


    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        labels = trainer.datamodule.hparams.labels
        dim_order = trainer.datamodule.hparams.dim_order
        data_dir = trainer.datamodule.hparams.data_dir
        im_channels = trainer.datamodule.hparams.im_channels
        sep = trainer.datamodule.hparams.sep
        ext = trainer.datamodule.hparams.ext

        logits,file_ids = outputs
        pred_masks = self.mask_regions_to_labels(data_dir, logits, labels, dim_order)
        N, W, H, D = pred_masks.shape
        for i in range(N):
            pred_mask = pred_masks[i].cpu().numpy()
            file_id = file_ids[i]
            affine, header = self.load_affine_and_header(data_dir,file_id,im_channels,sep,ext)
            pred_mask = nibabel.nifti1.Nifti1Image(pred_mask, affine, header=header)
            nibabel.save(pred_mask, os.path.join(self.save_dir, file_id + ".nii.gz"))


    def on_predict_start(self, trainer, pl_module):
        os.makedirs(self.save_dir,exist_ok=True)



class MultipleMasksSaveCallBack(PredictedMasksSaveCallBack):
    def __init__(self, save_dir):
        super().__init__(save_dir)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        labels = trainer.datamodule.hparams.labels
        dim_order = trainer.datamodule.hparams.dim_order
        data_dir = trainer.datamodule.hparams.data_dir
        im_channels = trainer.datamodule.hparams.im_channels
        sep = trainer.datamodule.hparams.sep
        ext = trainer.datamodule.hparams.ext

        pred_masks_dict, file_ids = outputs

        for key in pred_masks_dict.keys():
            out_dir = os.path.join(self.save_dir,key)
            os.makedirs(out_dir,exist_ok=True)

            pred_masks = self.mask_regions_to_labels(data_dir, pred_masks_dict[key], labels, dim_order)
            N, W, H, D = pred_masks.shape
            for i in range(N):
                pred_mask = pred_masks[i].cpu().numpy()
                file_id = file_ids[i]
                affine, header = self.load_affine_and_header(data_dir,file_id,im_channels,sep,ext,split='val')
                pred_mask = nibabel.nifti1.Nifti1Image(pred_mask, affine, header=header)
                nibabel.save(pred_mask, os.path.join(out_dir, file_id + ".nii.gz"))


    def on_test_start(self, trainer, pl_module):
        os.makedirs(self.save_dir,exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        labels = trainer.datamodule.hparams.labels
        dim_order = trainer.datamodule.hparams.dim_order
        data_dir = trainer.datamodule.hparams.data_dir
        im_channels = trainer.datamodule.hparams.im_channels
        sep = trainer.datamodule.hparams.sep
        ext = trainer.datamodule.hparams.ext

        pred_masks_dict, file_ids = outputs

        for key in pred_masks_dict.keys():
            out_dir = os.path.join(self.save_dir,key)
            os.makedirs(out_dir,exist_ok=True)

            pred_masks = self.mask_regions_to_labels(data_dir, pred_masks_dict[key], labels, dim_order)
            N, W, H, D = pred_masks.shape
            for i in range(N):
                pred_mask = pred_masks[i].cpu().numpy()
                file_id = file_ids[i]
                affine, header = self.load_affine_and_header(data_dir,file_id,im_channels,sep,ext,split='test')
                pred_mask = nibabel.nifti1.Nifti1Image(pred_mask, affine, header=header)
                nibabel.save(pred_mask, os.path.join(out_dir, file_id + ".nii.gz"))


    def on_predict_start(self, trainer, pl_module):
        os.makedirs(self.save_dir,exist_ok=True)