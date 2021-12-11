from typing import Dict, Union, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_metric_learning.losses import ArcFaceLoss


class BasicArcFaceSystem(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        n_class: int,
        embedding_dim: int,
        loss_args: Optional[Dict[str, Union[float, int]]] = None,
        model_optim_configs: Optional[Dict[str, Any]] = None,
        loss_optim_config: Optional[Dict[str, Any]] = None,
    ):
        super(BasicArcFaceSystem, self).__init__()
        self.model = model

        if not loss_args:
            loss_args = {
                "margin": 28.6,
                "scale": 64,
            }

        if not model_optim_configs:
            model_optim_configs = {
                "optim": optim.Adam,
                "args": {
                    "lr": 0.001,
                    "betas": (0.9, 0.99),
                    "eps": 1e-08,
                    "weight_decay": 0,
                    "amsgrad": False,
                }
            }

        if not loss_optim_config:
            loss_optim_config = {
                "optim": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0,
                    "dampening": 0,
                    "weight_decay": 0,
                    "nesterov": False,
                }
            }

        self.model_optim_config = model_optim_configs
        self.loss_optim_config = loss_optim_config

        self.criterion = ArcFaceLoss(
            num_classes=n_class, embedding_size=embedding_dim, **loss_args
        )

        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    def training_step(self, batch, batch_idx):
        model_optim, loss_optim = self.optimizers()
        model_optim.zero_grad()

        x, y = batch
        batch_size = x.shape[0]

        embedding = self(x).reshape(batch_size, -1)

        loss = self.criterion(embedding, y)

        self.manual_backward(loss)
        model_optim.step()
        loss_optim.step()

        self.log("train_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]

        embedding = self(x).reshape(batch_size, -1)
        loss = self.criterion(embedding, y)

        self.log("val_loss", loss)

    def configure_optimizers(self):
        model_optim = self.model_optim_config["optim"](
            self.model.parameters(), **self.model_optim_config["args"]
        )

        loss_optim = self.loss_optim_config["optim"](
            self.criterion.parameters(), **self.loss_optim_config["args"]
        )

        return model_optim, loss_optim
