import glob
import os
import cv2
import pickle
from sklearn import model_selection
import torch
import pytorch_lightning as pl
from typing import Any, Dict, Tuple, Optional
import numpy as np
import albumentations as albu
from loguru import logger


def get_pre_transforms(h: int, w: int):
    return [
        albu.PadIfNeeded(
            min_height=h,
            min_width=w,
            always_apply=True,
            border_mode=0,
            value=255,
        ),
        albu.RandomCrop(height=h, width=w, always_apply=True),
    ]


def get_test_augmentation(h: int, w: int):
    return albu.Compose(get_pre_transforms(h, w))


def get_training_augmentation(h: int, w: int):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=1,
            border_mode=0,
        ),
    ]

    train_transform.extend(get_pre_transforms(h, w))

    return albu.Compose(train_transform)


class TestDataset(torch.utils.data.Dataset):
    SIZE: Tuple[int, int] = (100, 100)

    def __init__(self, root: str, encoder: str):
        self.root = root
        self.paths = glob.glob(os.path.join(root, "*.png"))
        self.paths = np.array(self.paths)
        self.encoder = pickle.load(open(encoder, "rb"))
        self.transforms = get_test_augmentation(self.SIZE[0], self.SIZE[1])

    def __len__(self):
        return len(self.paths)

    def get_name(self, item):
        return os.path.sep.join(self.paths[item].rsplit(os.path.sep, 3)[-3:])

    def get_class_by_idx(self, idx):
        return self.encoder.inverse_transform([idx])

    def __getitem__(self, item):
        path = self.paths[item]
        image = cv2.imread(path, -1)
        if self.transforms is not None:
            sample = self.transforms(image=image)
            image = sample["image"]
        image = torch.from_numpy(image)[None]
        return {
            "image": image,
        }


class Dataset(torch.utils.data.Dataset):
    SIZE: Tuple[int, int] = (100, 100)

    def __init__(self, paths: np.ndarray, encoder: str, is_test: bool = False):
        super().__init__()
        self.paths = paths
        self.encoder = pickle.load(open(encoder, "rb"))
        self.transforms = (
            get_training_augmentation(self.SIZE[0], self.SIZE[1])
            if not is_test
            else get_test_augmentation(self.SIZE[0], self.SIZE[1])
        )

    def __len__(self):
        return len(self.paths)

    def _read(self, item: int) -> Tuple[np.ndarray, int]:
        path = self.paths[item]
        cls = os.path.split(path)[0]
        cls = os.path.split(cls)[-1]
        label = self.encoder.transform([cls])[0]
        image = cv2.imread(path, -1)
        return image, label

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        image, label = self._read(item)
        if self.transforms is not None:
            sample = self.transforms(image=image)
            image = sample["image"]
        image = torch.from_numpy(image)[None]
        return {
            "image": image,
            "label": label,
        }


class TrainDatamodule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.config = config
        self.train = None
        self.valid = None
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self) -> None:
        all_paths = glob.glob(
            os.path.join(self.config["dataset_path"], "*/*.png")
        )
        self.train, self.valid = model_selection.train_test_split(
            all_paths, test_size=0.2, random_state=42
        )
        self.valid = np.array(self.valid)
        self.train = np.array(self.train)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Dataset(self.train, self.config["encoder"])
        self.valid_dataset = Dataset(
            self.valid, self.config["encoder"], is_test=True
        )

        logger.info("Train stat: {}", len(self.train_dataset))
        logger.info("Valid stat: {}", len(self.valid_dataset))

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.config["valid"]["batch_size"],
            shuffle=False,
            num_workers=self.config["valid"].get("num_workers", 10),
            persistent_workers=True,
            pin_memory=True,
        )
        return dataloader

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 10),
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader
