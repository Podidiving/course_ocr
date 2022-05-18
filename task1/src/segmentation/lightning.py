import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics
from typing import Any, Dict


class LightningModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model = smp.create_model(
            config["arch"],
            encoder_name=config["encoder_name"],
            in_channels=config["in_channels"],
            classes=config["classes"],
        )

        params = smp.encoders.get_preprocessing_params(config["encoder_name"])
        self.register_buffer(
            "std", torch.tensor(params["std"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)
        )

        self.dice_loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )
        self.focal_loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        self.jaccard_loss_fn = smp.losses.JaccardLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )

        self.save_hyperparameters(config)

        self.iou = metrics.IoU(threshold=0.5)

    def forward(self, x: torch.Tensor):
        image = (x - self.mean) / self.std
        mask = self.model(image)
        return mask

    def _calc_loss(self, pred, mask):
        return {
            "focal": self.focal_loss_fn(pred, mask),
            "dice": self.dice_loss_fn(pred, mask),
            "jaccard": self.jaccard_loss_fn(pred, mask),
        }

    def _step(self, batch, stage: str):
        image = batch["image"]
        mask = batch["mask"]

        pred = self(image)
        losses = self._calc_loss(pred, mask)
        loss = losses["focal"] + losses["dice"] + losses["jaccard"]

        iou = self.iou(pred[:, 0], mask)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
        self.log(
            f"{stage}_focal_loss", losses["focal"], on_step=True, on_epoch=True
        )
        self.log(
            f"{stage}_dice_loss", losses["dice"], on_step=True, on_epoch=True
        )
        self.log(
            f"{stage}_jaccard_loss",
            losses["jaccard"],
            on_step=True,
            on_epoch=True,
        )

        return {"loss": loss, "iou": iou}

    def _epoch_end(self, outputs, stage):
        iou = torch.cat([x["iou"][None] for x in outputs]).mean()

        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_index):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_index):
        return self._step(batch, "val")

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.hparams.monitor_mode,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.hparams.monitor_metric,
            },
        }
