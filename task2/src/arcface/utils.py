from sklearn import preprocessing
import pickle
import glob
import numpy as np
import os
from typing import Any, Dict
from loguru import logger


def dump_encoder(config: Dict[str, Any]) -> None:
    if os.path.isfile(config["datamodule"]["encoder"]):
        logger.info("{} already exists.", config["datamodule"]["encoder"])
        return

    all_paths = glob.glob(
        os.path.join(config["datamodule"]["dataset_path"], "*/*.png")
    )
    logger.info("Total images: {}", len(all_paths))
    encoder = preprocessing.LabelEncoder()
    labels = np.array(
        list(set([os.path.split(os.path.split(x)[0])[-1] for x in all_paths]))
    )
    encoder.fit(labels)

    pickle.dump(encoder, open(config["datamodule"]["encoder"], "wb"))
