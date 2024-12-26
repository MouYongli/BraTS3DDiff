import torch.nn as nn

import torch

class MultiResPatchClassifyLoss(nn.Module):
    def __init__(self, mode="classify", patch_res=[16, 32], scale_loss=1.0):
        super().__init__()
        assert mode in ["classify", "regress"]
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.patch_res = patch_res
        self.scale_loss = scale_loss

    def _loss_mse(self, p, y):
        return self.mse(p, y.float())

    def _loss_bce(self, p, y):
        return self.bce(p, y.float())

    def forward(self, patch_preds:dict[torch.Tensor], patch_trues:dict[torch.Tensor]):
        assert len(patch_preds) == len(patch_trues) == len(self.patch_res) > 1

        # num_channels==1
        assert (
            list(patch_preds.values())[0].shape[1]
            == list(patch_trues.values())[0].shape[1]
            == 1
        )

        if self.mode == "regress":
            loss = {"loss": 0.0}
            loss.update({f"loss_res={k}": 0.0 for k in self.patch_res})

            for patch_res in self.patch_res:
                # for every patch resolution
                patch_pred = patch_preds[patch_res]
                patch_true = patch_trues[patch_res]
                mse_loss = self._loss_mse(patch_pred, patch_true)
                loss[f"mse_loss_res={patch_res}"] = mse_loss
                loss["loss"] += loss[f"mse_loss_res={patch_res}"]
            return loss

        elif self.mode == "classify":
            loss = {"patch_classify_loss": 0.0}

            for patch_res in self.patch_res:
                # for every patch resolution
                patch_pred = patch_preds[patch_res]
                patch_true = patch_trues[patch_res]

                loss[f"patch_classify_bce_loss_res={patch_res}"] = self._loss_bce(patch_true, patch_true)
                loss["patch_classify_loss"] += loss[
                    f"patch_classify_bce_loss_res={patch_res}"
                ]

            loss["patch_classify_loss"] *= self.scale_loss

            return loss
