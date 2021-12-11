from typing import Dict, Union, Optional
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
    ):
        super(BasicArcFaceSystem, self).__init__()
        self.model = model

        if not loss_args:
            loss_args = {
                "margin": 28.6,
                "scale": 64,
            }

        self.criterion = ArcFaceLoss(
            num_classes=n_class, embedding_size=embedding_dim, **loss_args
        )

        self.automatic_optimization = False

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

    def configure_optimizers(self):
        model_optim = optim.Adam(self.model.parameters())
        loss_optim = optim.SGD(self.criterion.parameters(), lr=0.01)
        return model_optim, loss_optim
