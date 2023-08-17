import os
import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

if __name__ == "__main__":
    dm = MNISTDataModule(batch_size=32)
    model = LitClassifier()
    env = LightningEnvironment()
    env.world_size = lambda: int(os.environ.get("WORLD_SIZE", 0))
    env.global_rank = lambda: int(os.environ.get("RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_nodes = int(world_size/num_gpus)
    ddp = DDPPlugin(
        parallel_devices=[torch.device("cuda", d) for d in range(num_gpus)],
        cluster_environment=env)
    trainer = pl.Trainer(gpus=num_gpus, num_nodes=num_nodes, max_epochs=200, strategy=ddp)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
