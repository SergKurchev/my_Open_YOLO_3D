import os
import fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from openyolo3d.module import OpenYolo3DModule
from openyolo3d.datamodule import OpenYolo3DDataModule
from loguru import logger

def train(
    data_path: str,
    config_path: str = "./pretrained/config_scannet200.yaml",
    splits_path: str = None,
    epochs: int = 50,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    name: str = "openyolo3d_modernized",
    use_wandb: bool = False,
    debug: bool = False
):
    """
    Main training function using PyTorch Lightning.
    """
    # 1. Setup DataModule
    dm = OpenYolo3DDataModule(
        data_path=data_path,
        splits_path=splits_path,
        batch_size=batch_size
    )
    
    # 2. Setup LightningModule
    model = OpenYolo3DModule(
        config_path=config_path,
        lr=lr,
        weight_decay=weight_decay
    )
    
    # 3. Setup Loggers
    loggers = [CSVLogger("logs", name=name)]
    if use_wandb:
        loggers.append(WandbLogger(project="OpenYOLO3D", name=name))
    
    # 4. Setup Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{name}",
            filename="best-{epoch:02d}-{val_status:.2f}",
            save_top_k=1,
            monitor="val_status",
            mode="max",
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # 5. Setup Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=callbacks,
        fast_dev_run=debug,
        log_every_n_steps=1 if debug else 10
    )
    
    # 6. Start Training
    logger.info(f"Starting training: {name}")
    trainer.fit(model, datamodule=dm)
    logger.info("Training finished!")

if __name__ == "__main__":
    fire.Fire(train)
