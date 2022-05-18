# just a copy
import zipfile
from typing import Optional, List
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict, Counter


class Vocabulary:
    def __init__(self, classes):
        self.classes = sorted(set(classes))
        self._class_to_index = dict((cls, idx) for idx, cls in enumerate(self.classes))
    
    def class_by_index(self, idx: int) -> str:
        return self.classes[idx]

    def index_by_class(self, cls: str) -> int:
        return self._class_to_index[cls]
    
    def num_classes(self) -> int:
        return len(self.classes)


class ArchivedHWDBReader:
    def __init__(self, path: Path):
        self.path = path
        self.archive = None
    
    def open(self):
        self.archive = zipfile.ZipFile(self.path)
    
    def namelist(self):
        return self.archive.namelist()
    
    def decode_image(self, name):
        sample = self.archive.read(name)
        buf = np.asarray(bytearray(sample), dtype='uint8')
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    
    def close(self):
        self.archive.close()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class HWDBDatasetHelper:
    def __init__(self, reader, prefix='Train', vocabulary: Optional[Vocabulary]=None, namelist: Optional[List[str]]=None):
        self.reader = reader
        self.prefix = prefix
        self.index = defaultdict(list)
        self.counter = Counter()
        self.namelist = namelist
        if self.namelist is None:
            self.namelist = list(filter(lambda x: self.prefix in x, self.reader.namelist()))
        self.vocabulary = vocabulary
        self._build_index()
    
    def get_item(self, idx):
        name = self.namelist[idx]
        return self.reader.decode_image(name), \
            self.vocabulary.index_by_class(HWDBDatasetHelper._get_class(name))
    
    def size(self):
        return len(self.namelist)

    def get_all_class_items(self, idx):
        cls = self.vocabulary.class_by_index(idx)
        return self.index[cls]
    
    def most_common_classes(self, n=None):
        return self.counter.most_common(n)
    
    def train_val_split(self, train_part=0.8, seed=42):
        rnd = np.random.default_rng(seed)
        permutation = rnd.permutation(len(self.namelist))
        train_part = int(len(permutation) * train_part)
        train_names = [self.namelist[idx] for idx in permutation[:train_part]]
        val_names = [self.namelist[idx] for idx in permutation[train_part:]]

        return HWDBDatasetHelper(self.reader, self.prefix, self.vocabulary, train_names),\
            HWDBDatasetHelper(self.reader, self.prefix, self.vocabulary, val_names)
    
    @staticmethod
    def _get_class(name):
        return Path(name).parent.name
    
    def _build_index(self):
        classes = set()
        for idx, name in enumerate(self.namelist):
            cls = HWDBDatasetHelper._get_class(name)
            classes.add(cls)
            self.index[cls].append(idx)
            self.counter.update([cls])
        
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(classes)
