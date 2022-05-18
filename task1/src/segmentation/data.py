import torch
import pytorch_lightning as pl
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
import albumentations as albu
from loguru import logger

from course_ocr_t1.data import MidvPackage


def get_pre_transforms(h: int, w: int):
    return [
        albu.PadIfNeeded(
            min_height=h, min_width=w, always_apply=True, border_mode=0
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


class Dataset(torch.utils.data.Dataset):
    BASE: int = 320
    ORIGINAL_SHAPE: Tuple[int, int] = (800, 450)

    def __init__(self, path: str, is_test: bool = False):
        super().__init__()
        self.path = Path(path)
        self.is_test = is_test
        self.images = None
        self.masks = None
        self._init()
        self.transforms = (
            get_training_augmentation(self.BASE * 3, self.BASE * 2)
            if not is_test
            else get_test_augmentation(self.BASE * 3, self.BASE * 2)
        )

    def _init(self):
        data_packs = MidvPackage.read_midv500_dataset(self.path)
        self.images = []
        self.masks = []
        for pack in data_packs:
            for di in pack:
                if self.is_test == di.is_test_split() and di.is_correct():
                    self.images.append(di.img_path)
                    self.masks.append(np.array(di.gt_data["quad"]))
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

    @staticmethod
    def __make_mask(shape: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, np.float32)
        cv2.fillPoly(mask, np.array([poly.tolist()]), 1)
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        image = cv2.imread(str(self.path / self.images[item]))[:, :, ::-1]
        mask = self.__make_mask(image.shape[:2], self.masks[item])
        if self.transforms is not None:
            sample = self.transforms(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]
        image = (
            torch.permute(torch.from_numpy(image).float(), (2, 0, 1)) / 255.0
        )
        mask = torch.from_numpy(mask).float()
        return {
            "image": image,
            "mask": mask,
        }


class Datamodule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dataset_path = config["dataset_path"]
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Dataset(self.dataset_path, is_test=False)
        self.valid_dataset = Dataset(self.dataset_path, is_test=True)
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
        )
        return dataloader
