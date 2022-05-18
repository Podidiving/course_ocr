from argparse import ArgumentParser, Namespace
from typing import Any, Dict
import os
import yaml
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import time

from .lightning import LightningModel
from .data import TrainDatamodule
from .utils import dump_encoder


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dict", type=str, required=False, default="dict.yaml"
    )
    return parser.parse_args()


def read(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def main(config: Dict[str, Any]) -> None:
    dump_encoder(config)
    logger.info("Training is initialized")
    pl.seed_everything(42)

    log_path = os.path.join(
        config["logging"]["logdir"],
        config["logging"]["experiment_name"],
        str(int(time.time())),
    )
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    callbacks = [
        ModelCheckpoint(
            dirpath=log_path,
            filename=config["logging"]["experiment_name"] + "-{epoch:02d}",
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
            mode="max",
            auto_insert_metric_name=False,
        ),
    ]

    loggers = [
        TensorBoardLogger(
            save_dir=log_path,
            name="",
        ),
    ]
    datamodule = TrainDatamodule(config["datamodule"])
    module = LightningModel(config["module"])

    trainer = pl.Trainer(
        **config["trainer"],
        logger=loggers,
        callbacks=callbacks,
    )
    logger.info("Training preparation process is done.")
    trainer.fit(module, datamodule=datamodule)
    trainer.save_checkpoint(os.path.join(log_path, "final.ckpt"))
    logger.info("Training is done.")


if __name__ == "__main__":
    args = parse_args()
    config_ = read(args.dict)
    main(config_)
