import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics import Accuracy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import wandb

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, StepLR
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

steps = 20_000

class DeepFakeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='dataset', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_ds = datasets.ImageFolder(self.data_dir + '/train', transform=self.transform)
        self.val_ds = datasets.ImageFolder(self.data_dir + '/val', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

class DeepFakeDetector(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Use pretrained ResNet18
        # self.model = models.resnet18(pretrained=True)
        self.model = models.resnet34(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model.fc = nn.Identity()

        
        # Initialize metrics and loss
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)
        self.log_dict({'train_loss': loss, 'train_acc': self.train_acc}, 
                     on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.log_dict({'val_loss': loss, 'val_acc': self.val_acc}, 
                     on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=3, gamma=0.5), #0.1),  # Example: StepLR
            'interval': 'epoch',  # 'epoch' or 'step'
            'frequency': 1,       # How often to apply the scheduler
            'reduce_on_plateau': False,  # For ReduceLROnPlateau
            'monitor': 'val_loss',  # Metric to monitor for ReduceLROnPlateau
        }

        return [optimizer], [scheduler]

if __name__ == '__main__':
    wandb.init(
        project='snicket',
    )
    logger = WandbLogger()

    # Initialize data and model
    dm = DeepFakeDataModule(data_dir='data', batch_size=64)
    model = DeepFakeDetector(learning_rate=1e-4)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",          # Directory to save checkpoints
        filename="best-model-{epoch:02d}-{val_loss:.2f}",  # Checkpoint file name
        monitor="val_loss",              # Metric to monitor
        mode="min",                      # Minimize the monitored metric
        save_top_k=1,                    # Save only the best model
        save_last=True,                  # Optionally save the last checkpoint
    )

    # Train the model
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        deterministic=True,
        logger=logger,
        max_steps=steps,
        callbacks=[lr_monitor]
    )
    
    trainer.fit(model, datamodule=dm)