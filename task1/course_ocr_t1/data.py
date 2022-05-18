from dataclasses import dataclass
import json
from pathlib import Path
from typing import List
import warnings

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image

from .metrics import iou_relative_quads


@dataclass
class DataItem:
    """
    Изображение с разметкой-четырехугольником кропа.
    """
    gt_path: Path
    img_path: Path

    @property
    def unique_key(self):
        return "|".join(self.gt_path.parts[-4:])

    @property
    def image(self) -> Image:
        return Image.open(self.img_path)

    @property
    def gt_data(self) -> dict:
        if hasattr(self, '_gt_data'):
            return self._gt_data

        try:
            with open(self.gt_path, 'r') as f:
                self._gt_data = json.loads(f.read())
        except UnicodeDecodeError as exc:
            # в оригинальных пакетах есть в шаблонах текст в не UTF-8 кодировках, например
            # китайский; этот текст не нужен для задания
            warnings.warn(f"Can't read {self.gt_path}, non unicode text", UnicodeWarning)
            self._gt_data = dict()
        return self._gt_data

    @property
    def quadrangle(self) -> np.array:
        gt_data = self.gt_data
        if 'quad' in gt_data:
            quad = np.array(self.gt_data['quad']).astype(float)
            quad /= self.image_size[None]
            return quad
        return None

    @property
    def image_size(self):
        if not hasattr(self, '_image_size'):
            self._image_size = np.array(self.image.size).astype(float)
        return self._image_size

    def is_correct(self) -> bool:
        """
        Проверяет, существуют ли файлы изображения и разметки, и если да,
        то есть ли в файле разметки данные четырехугольника.
        Для разметки 'шаблона', т.е. идеально кропнутого изображения, лежащего в корне,
        этих дданных нет, потому что quad = [(0,0), (0,w), (h,w), (0,h)].
        """
        exists = (self.gt_path.exists()) and (self.img_path.exists())
        return exists and ('quad' in self.gt_data)

    def show(self, quad01=None):
        """
        Отрисовывает четырехугольник кропа поверх изображения,
        либо quad01, либо quadrangle из разметки.
        quad01 должен быть в относительно масштабе (min ~ 0, max ~ 1).
        """
        fig, ax = plt.subplots()
        ax.imshow(np.array(self.image))
        if quad01 is None:
            quad01 = self.quadrangle
        quad = quad01 * self.image_size[None]
        p = patches.Polygon(quad, edgecolor = 'r', fill=False)
        ax.add_patch(p)

    def is_quad_inside(self) -> bool:
        """
        Находится ли четырехугольник разметки полностью внутри кадра.
        """
        assert self.is_correct(), "Incorrect item quad is undefined"
        return np.all(0 <= self.quadrangle) and np.all( self.quadrangle <= 1)

    def is_test_split(self) -> bool:
        """
        Принадлежит ли итем к test-разделу датасета.
        Правило разделения train/test следующее:
        - если итем принадлежит пакету с номером [44:50], то это test split
        - если итем в последовательности имеет номера [26:30], то это test split
        Таким образом, в тесте есть:
        ~ 7 * 300=2100 итемов, для которых не было примеров во время тренировки
        ~ 43 * (5/30) * 10=2150 итемов, чей которых были примеры во время тренировки
        Всего 4150 итемов.
        """
        assert self.is_correct(), "Incorrect item split is undefined"
        package_num = int(self.gt_path.parent.parent.parent.name.split('_')[0])
        sequence_num = int(self.gt_path.stem.split('_')[-1])

        if 25 < sequence_num <= 30:
            return True

        if 43 < package_num <= 50:
            return True
        return False

    def iou_with(self, pred_quad01: np.array) -> float:
        assert self.is_correct(), "Can't measure iou for incorrect item"
        return iou_relative_quads(self.quadrangle, pred_quad01)

    def __repr__(self):
        return f"DataItem<{self.gt_path}|{self.img_path}>"


class MidvPackage:
    """
    Один пакет данных MIDV-500.
    Контейнер для DataItem, лежащих в одном корневом
    """
    GT_SUBDIR = 'ground_truth'
    IMG_SUBDIR = 'images'

    @classmethod
    def read_midv500_dataset(cls, dataset_root:Path)->List['MidvPackage']:
        """Читает весь датасет MIDV-500 (список из пакетов 50)"""
        if not isinstance(dataset_root, Path):
            raise TypeError(f"Expected pathlib.Path type, got {type(dataset_root)}")
        package_paths = [p for p in dataset_root.glob("*") if p.is_dir()]
        return [cls(p) for p in package_paths]

    def __init__(self, root: Path):
        for subdir in (self.GT_SUBDIR, self.IMG_SUBDIR):
            assert (root / subdir).exists(), f"{self.root} does not contain '{subdir}'"

        self.root = root
        self.items = self.collect_items(root)
        self.template_item = self.collect_template(root)

    @classmethod
    def collect_items(cls, root_path: Path) -> List[DataItem]:
        """
        Собирает все DataItem в пакете, кроме шаблона.
        """
        items = []
        gt_root = root_path / cls.GT_SUBDIR
        img_root = root_path / cls.IMG_SUBDIR

        gt_paths = list(gt_root.rglob("*.json"))
        img_paths = [x for x in list(img_root.rglob("*")) if not x.is_dir() and not '.ipynb' in str(x)]

        key = lambda x: str(x.parent / x.stem)
        key_to_img, key_to_gt = dict(), dict()
        for p in gt_paths:
            key_to_gt[key(p.relative_to(gt_root))] = p
        for p in img_paths:
            key_to_img[key(p.relative_to(img_root))] = p

        simg = set(key_to_img.keys())
        sgt =  set(key_to_gt.keys())
        assert simg == sgt, f"{simg - sgt}, {sgt - simg}"
        for k, img_path in key_to_img.items():
            gt_path = key_to_gt[k]
            items.append(DataItem(gt_path=gt_path, img_path=img_path))
        return [it for it in items if it.is_correct()]

    @classmethod
    def collect_template(cls, root_path: Path) -> DataItem:
        """
        Возвращает DataItem-шаблон пакета. Не
        """
        gt_path = root_path / cls.GT_SUBDIR / (root_path.name + ".json")
        assert gt_path.exists(), f"Expected {gt_path} to be template gt, but it does not exist"

        img_path = None
        for ext in (".png", ".tif"):
            _img_path = root_path / cls.IMG_SUBDIR / (root_path.name + ext)
            if (_img_path.exists()):
                img_path = _img_path
        assert img_path is not None, "Expected to find template img, but it does not exist"
        return DataItem(gt_path=gt_path, img_path=img_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __repr__(self):
        return f"DataPackage[{self.root}]({len(self)})"
