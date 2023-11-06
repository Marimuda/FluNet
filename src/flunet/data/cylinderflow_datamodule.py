"""This module defines the `MeshDataModule` class, which is a PyTorch Lightning data module for loading and processing
mesh data. The `MeshDataModule` class provides a standardized interface for loading and processing mesh data, and can be
easily integrated into any PyTorch Lightning project that requires mesh data.

The `MeshDataModule` class inherits from the `pl.LightningDataModule` class, and provides several methods for preparing
and loading data, including `prepare_data()`, `setup()`, `train_dataloader()`, and `val_dataloader()`. The
`MeshDataModule` class also provides a `transfer_batch_to_device()` method for transferring batches of data to the GPU
for training.

Overall, the `MeshDataModule` class provides a convenient and standardized way to load and process mesh data for use in
PyTorch Lightning models.
"""
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from flunet.data.dataset.meshdataset import MeshDataset


class MeshDataModule(LightningDataModule):
    """This class defines the data module for the Mesh dataset."""

    def __init__(self, transform=None, **kwargs):
        super().__init__()
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """This method is used to download and prepare the data."""
        print("Preparing data...")

    def setup(self, stage: Optional[str] = None):
        """This method is used to split the data into train and validation sets."""
        print("Setting up dataset...")
        if stage == "fit" or stage is None:
            self.data_train = MeshDataset(self.hparams, split="train")
            self.data_val = MeshDataset(self.hparams, split="valid")
            print("Number of training data: ", len(self.data_train))
            print("Number of validation data: ", len(self.data_val))

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        """This method is used to transfer the batch to the specified device."""
        batch = batch.to(device)
        return batch

    def train_dataloader(self):
        """This method returns the training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        """This method returns the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


if __name__ == "__main__":
    datamodule = MeshDataModule(
        transform=lambda x: x,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        dataset_name="cylinder_flow",
        data_dir="./data/cylinder_flow",
        noise_scale=0.02,
        noise_gamma=1.0,
        field="velocity",
        history=False,
    )
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule.train_dataloader())
    print(datamodule.val_dataloader())
