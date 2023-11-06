import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import logging
from datamodules.dataset_BrainMRI import BrainDataset
from pytorch_lightning.utilities import rank_zero_info
from helpers import load_json
from define_parameters import Parameters
from pathlib import Path

txt_logger = logging.getLogger("pytorch_lightning")


def filter_brainslices(
    datapath: Path, slice_indices: list, train_names: list[Path], train_labels: list[Path], data_part: float = 1.0
) -> tuple[list[Path], list[Path]]:
    """Function to filter the brain slices to only include the slices that are in the slice_indices list

    :param datapath: path where the data is located
    :param slice_indices: slice indices to include
    :param train_names: names of the volumes to include
    :param train_labels: labels of the volumes to include
    :param data_part: fraction of the data to use
    :return: filteres trainslices and corresponding labels
    """
    train_files = []
    filtered_labels = []
    for num, name in enumerate(train_names):
        for slice in slice_indices:
            filepath = datapath / name / f"slice_{slice}.png"
            train_files.append(filepath)
            filtered_labels.append(train_labels[num])

    num_files = int(len(train_files) * data_part)
    train_files = train_files[:num_files]
    filtered_labels = filtered_labels[:num_files]

    return train_files, filtered_labels


class MyDataModuleBrainDataset(pl.LightningDataModule):
    def __init__(self, params: Parameters):
        super().__init__()
        self.params = params

        # Load datadict
        datadict = load_json(params.datasplit_path)

        slice_indices_train = self.params.slice_indices_train
        slice_indices_test = self.params.slice_indices_test

        # extract correct train files -------------------------------------------------
        train_names = datadict[f"Fold_{params.cv_fold}"]["train"]["files"]
        train_labels = datadict[f"Fold_{params.cv_fold}"]["train"]["ages"]

        self.train_files, self.train_labels = filter_brainslices(
            Path(self.params.data_path),
            slice_indices=slice_indices_train,
            train_names=train_names,
            train_labels=train_labels
        )

        # extract correct val files -------------------------------------------------------
        val_names = datadict[f"Fold_{params.cv_fold}"]["val"]["files"]
        val_labels = datadict[f"Fold_{params.cv_fold}"]["val"]["ages"]

        self.val_files, self.val_labels = filter_brainslices(
            Path(self.params.data_path),
            slice_indices=slice_indices_train,
            train_names=val_names,
            train_labels=val_labels
        )

        # get test files ------------------------------------------------------
        test_names = datadict["test"]["files"]
        test_labels = datadict["test"]["ages"]

        self.test_files, self.test_labels = filter_brainslices(
            Path(self.params.data_path),
            slice_indices=slice_indices_test,
            train_names=test_names,
            train_labels=test_labels
        )

        self.min_train_label = min(self.train_labels)
        self.max_train_label = max(self.train_labels)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Dataset required for training

        Returns:
            pytorch dataloader
        """

        # Data augmentation
        augm_transform = transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), translate=(0.02, 0.02))

        train_dataset = BrainDataset(
            self.train_files,
            self.train_labels,
            preload=self.params.preload_data,
            transform=augm_transform,
        )

        rank_zero_info(f"Training dataset size: {len(train_dataset)}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.params.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader

    def push_dataloader(self) -> torch.utils.data.DataLoader:
        """Dataset required for prototype pushing

        Returns:
            pytorch dataloader
        """
        push_dataset = BrainDataset(self.train_files, self.train_labels, preload=self.params.preload_data)

        rank_zero_info(f"Pushing dataset size: {len(push_dataset)}")
        push_loader = torch.utils.data.DataLoader(
            push_dataset,
            batch_size=self.params.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        return push_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Dataset required for validation

        Returns:
            pytorch dataloader
        """
        val_dataset = BrainDataset(self.val_files, self.val_labels, preload=self.params.preload_data)

        rank_zero_info(f"Validation dataset size: {len(val_dataset)}")
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.params.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        return val_loader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Dataset required for testing

        Returns:
            pytorch dataloader
        """
        test_dataset = BrainDataset(self.test_files, self.test_labels, preload=self.params.preload_data)

        rank_zero_info(f"Testing dataset size: {len(test_dataset)}")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.params.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        return test_loader
