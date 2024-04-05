import pytorch_lightning as pl

from model import GAN  
import config
from data import DataModule
from torchvision import transforms
import wandb
from pytorch_lightning.strategies import DDPStrategy


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Subtract 0.5 from each channel to shift to [-0.5, 0.5]
                         std=[0.5, 0.5, 0.5])   # Divide by 0.5 to scale to [-1, 1]
])
dm = DataModule(config.TRAIN_PATH, config.TEST_PATH, config.BATCH_SIZE, config.NUM_WORKERS, transform,config.DEVICE)
strategy = DDPStrategy(find_unused_parameters=True)
model = GAN(learning_rate=config.LR, z_dim=config.Z_DIM)
trainer = pl.Trainer(accelerator="gpu", devices=[0], strategy=strategy, max_epochs=config.EPOCHS)
# trainer = pl.Trainer(max_epochs=config.EPOCHS, strategy=strategy)
wandb.init(project='DC-GAN')
trainer.fit(model, dm)