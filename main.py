import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics import Accuracy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import wandb
from pytorch_lightning.callbacks import Callback

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
        self.log_dict({'train/loss': loss, 'train/acc': self.train_acc}, 
                     on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.log_dict({'val/loss': loss, 'val/acc': self.val_acc}, 
                     on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.2)
    
class SaveModelCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_score = 0.0  # Initialize best score to zero

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the current validation accuracy
        if(trainer.current_epoch > 5):
            current_score = trainer.callback_metrics["val/acc"].item()

            # Save the model if validation accuracy improves
            if current_score > self.best_score:
                self.best_score = current_score
                best_model_path = "deepfake_detector_best_moe.pth"
                torch.save(pl_module.model.state_dict(), best_model_path)
                print(f"New best model saved at epoch {trainer.current_epoch + 1} with val/acc: {current_score:.4f}")


if __name__ == '__main__':
    wandb.init(
        project='snicket',
    )
    logger = WandbLogger()

    # Initialize data and model
    dm = DeepFakeDataModule(data_dir='data', batch_size=64)
    model = DeepFakeDetector(learning_rate=1e-5)
    save_model_callback = SaveModelCallback()
    # Train the model
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        deterministic=True,
        logger=logger
        callbacks=[save_model_callback]
    )
    
    trainer.fit(model, datamodule=dm)