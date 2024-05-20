import torch as th
import lightning as L
import os
from models.utils import logger
from models.utils.utils import params_to_state_dict
import blobfile as bf


class ModelSaveCallback(L.Callback):
    '''
    ModelSaveCallback
    '''

    def __init__(self, save_dir:str,save_interval:int=10000) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_interval = save_interval

    def save(self,trainer: L.Trainer, pl_module:L.LightningModule):
        def save_checkpoint(rate, params):
            state_dict = params_to_state_dict(pl_module.net,params)
            #logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(trainer.global_step+pl_module.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(trainer.global_step+pl_module.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, pl_module.net_params)
        for rate, params in zip(pl_module.ema_rate, pl_module.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(trainer.global_step+pl_module.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(pl_module.optimizers().optimizer.state_dict(), f)


    def on_train_batch_end(self, trainer: L.Trainer, pl_module:L.LightningModule, outputs, batch, batch_idx) -> None:
        # do something with all training_step outputs, for example:

        if trainer.global_step % self.save_interval == 0:
            self.save(trainer,pl_module)

