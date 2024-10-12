import argparse
import os
import re
import shutil
from datetime import datetime

import nibabel as nib
import synapseclient
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from synapseclient import File

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SynapseSubmissionCallBack(Callback):
    def __init__(
        self,
        auth_file,
        syn_proj_id,
        pred_labels_path,
        eval_images_path,
        syn_eval_id="9615339",
        zip_file_name=None,
        name=None,
        team=None,
    ):
        """Implementation of Synapse submission callback for BRATS-2023-GLI.

        Parameters:
        - auth_file: Path to the authentication token file. Can be obtained from Synapse website for approved users.
        - syn_proj_id: Synapse object ID of the project to which the file has to be added. Corresponds to the Synapse ID of the project created which is also enrolled in the competition.
        - pred_labels_path: Path of the predicted segmentation masks
        - eval_images_path: Path of the original images for which the predicted labels are provided.
        - syn_eval_id: the ID of the evaluation queue that the competition uses for submission. Default is the ID for the Brain Tumor Segmentation Challenge.
        - zip_file_name: Name of the zip file to be created. If not provided, a default name will be used.
        - name: Name of the submission. If not provided, a default name will be used.
        """

        assert auth_file != None, "Authentication File Path Cannot be None"
        assert syn_proj_id != None, "Synapse Project ID Cannot be None"
        assert pred_labels_path != None, "Predicted labels path cannot be None"

        self.auth_file = auth_file
        self.syn_proj_id = syn_proj_id
        self.syn_eval_id = syn_eval_id
        self.pred_labels_path = pred_labels_path
        self.eval_images_path = eval_images_path
        self.zip_file_name = (
            f'submission_{datetime.now().strftime("%H_%M_%S")}'
            if not zip_file_name
            else zip_file_name
        )
        self.submission_file_name = self.zip_file_name + ".zip"
        self.syn_submission_name = (
            f"Submission_{datetime.now().strftime('%H_%M_%S')}" if not name else name
        )
        self.syn_team = team

    # Methods for validation checks before submission
    def correct_file_paths_dic(self):
        """Returns a set containing the correct file path."""
        assert os.path.exists(self.eval_images_path), "Eval Images Path Does Not Exists"
        directories = {
            "-".join(f.split("-")[-2:])
            for f in os.listdir(self.eval_images_path)
            if os.path.isdir(os.path.join(self.eval_images_path, f))
        }
        return directories

    def validate_file_names(self):
        """Validates that the file names are in the correct pattern.

        Returns:
        - bool: True if all files are in corrent format
        """

        assert os.path.exists(self.eval_images_path), "Eval Images Path Does Not Exists"
        assert os.path.exists(
            self.pred_labels_path
        ), "Predicted Segmentation Masks Path Does Not Exists"

        valid_ids = self.correct_file_paths_dic()
        files = os.listdir(self.pred_labels_path)

        # Validating each file name
        for file_name in files:
            match = re.match(
                r"^.*?([a-zA-Z0-9_\-]+)-([a-zA-Z0-9_\-]+)\.nii\.gz$", file_name
            )
            if match:
                entry = f"{match.group(1)}-{match.group(2)}"
                if entry[-9:] in valid_ids:
                    continue
                else:
                    print(f"Couldn't match {entry}")
                    return False
            else:
                print(f"Error in Name")
                return False

        return True

    def validate_masks_shape(self):
        """Validates the size of images in a given folder.

        This function checks whether all images in the specified folder have the
        dimensions (240, 240, 155).

        Args:
            folder_path (str): The path to the folder containing the images.

        Returns:
            bool: True if all images have the correct size, False otherwise.
        """
        files = os.listdir(self.pred_labels_path)
        for i, j in enumerate(files):
            img = nib.load(self.pred_labels_path + "/" + j).get_fdata()
            if img.shape != (240, 240, 155):
                return False
        return True

    def submit_file(self, authentication_key):
        """Submits a file to a Synapse evaluation queue.

        This function logs into Synapse using an authentication key, uploads a file,
        and submits it to a specified evaluation queue.

        Args:
            authentication_key (str): The authentication token for Synapse login.

        Returns:
            The submission object created on Synapse.
        """
        # Login to Synapse
        syn = synapseclient.login(authToken=authentication_key)
        # Upload the file to Synapse
        file = File(path=self.submission_file_name, parent=self.syn_proj_id)
        file = syn.store(file)

        # Submit the file to the evaluation queue
        submission = syn.submit(
            evaluation=self.syn_eval_id,
            entity=file.id,
            name=self.syn_submission_name,
        )  # Optional, can also pass a Team object or id

    @staticmethod
    def read_authentication_key(file_path):
        """Reads an authentication key from a file.

        This function opens a file from the given file path, reads the authentication key,
        and returns it after stripping any leading/trailing whitespace.

        Args:
            file_path (str): The path to the file containing the authentication key.

        Returns:
            str: The authentication key.
        """
        with open(file_path) as file:
            return file.read().strip()

    def validate_and_submit_file(self):
        assert self.validate_file_names(), "Error in Predicted labels file names"
        assert self.validate_masks_shape(), "Error in Predicted labels Shape"

        shutil.make_archive(self.zip_file_name, "zip", self.pred_labels_path)
        # Read the authentication key
        authentication_key = self.read_authentication_key(self.auth_file)
        self.submit_file(authentication_key)
        return

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule):
        log.info("Submitting predictions to Synapse... ")
        self.validate_and_submit_file()


if __name__ == "__main__":
    auth_file = "/home/sanyal/Projects/BraTS3DDiff/synapse_auth_keys/auth_key_rsanyal419_gmail.txt"
    syn_proj_id = "syn62281290"
    pred_labels_path = "/home/sanyal/Projects/BraTS3DDiff/logs/bratseg_logs/bratseg_baselines/BraTS23-SwinUNETR_baseline_0/predict/runs/2024-08-20_21-39-43/seg_masks"
    eval_images_path = (
        "/home/sanyal/Projects/BraTS3DDiff/data/BraTS-Data/BraTS2023-GLI/test"
    )
    syn_eval_id = "9615339"
    now = datetime.now()
    zip_file_name = f"BraTS23-SwinUNETR_baseline_0_{now:%Y-%m-%d}_{now:%H-%M-%S}"
    name = zip_file_name
    team = None
    syn_subm = SynapseSubmissionCallBack(
        auth_file=auth_file,
        syn_proj_id=syn_proj_id,
        pred_labels_path=pred_labels_path,
        eval_images_path=eval_images_path,
        syn_eval_id=syn_eval_id,
        zip_file_name=zip_file_name,
        name=name,
        team=team,
    )

    syn_subm.validate_and_submit_file()
