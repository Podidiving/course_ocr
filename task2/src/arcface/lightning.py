import torch
import pytorch_lightning as pl
from typing import Any, Dict

from .margin import ArcMarginProduct
from .model import ArcFace
from .focal import FocalLoss


class LightningModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.model = ArcFace(config["backbone"], config["feature_size"])
        self.margin = ArcMarginProduct(
            config["feature_size"], config["num_classes"]
        )
        self.criterion = (
            FocalLoss()
            if config.get("use_focal_loss", False)
            else torch.nn.CrossEntropyLoss()
        )

        self.save_hyperparameters(config)

    def forward_features(self, x: torch.Tensor):
        x = (x.float() - 127.5) / 255.0
        return self.model(x)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.margin(features)

    def _step(self, batch, stage: str):
        image = batch["image"]
        label = batch["label"]

        feature = self.forward_features(image)
        prediction = self.margin(feature, label)
        loss = self.criterion(prediction, label)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return {"loss": loss}

    def training_step(self, batch, batch_index):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_index):
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.margin.parameters())
        opt_name = self.hparams.optimizer["name"]
        opt_params = self.hparams.optimizer["params"]
        opt = torch.optim.__dict__[opt_name](params, **opt_params)
        if "lr_scheduler" in self.hparams:
            lr_params = self.hparams.lr_scheduler["params"]
        else:
            lr_params = {
                "step_size": 3,
                "gamma": 0.1,
            }
        sch = torch.optim.lr_scheduler.StepLR(opt, **lr_params)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
            },
        }
