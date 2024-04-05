import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from typing import Tuple

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Subtract 0.5 from each channel to shift to [-0.5, 0.5]
                         std=[0.5, 0.5, 0.5])   # Divide by 0.5 to scale to [-1, 1]
])
class CustomData(Dataset) :
  def __init__(self, path: str, transform: transforms.Compose) -> None:
    self.path = list(path.glob('*'))
    self.transform = transform
  def load_image(self, idx) -> Image.Image:
     image = Image.open(self.path[idx])
     return image
  def __len__(self) -> int:
    return len(self.path)
  def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
     image = self.transform(self.load_image(idx))
     return image

class DataModule(pl.LightningDataModule) :
  def __init__(self, train_dir: str, test_dir: str, batch_size: int, num_workers: int, transform: transforms.Compose, device) :
    super().__init__()
    self.train_dir = train_dir
    self.test_dir = test_dir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.transform = transform
  def prepare_data(self) :
    CustomData(self.train_dir, self.transform)
    CustomData(self.test_dir, self.transform)

  def setup(self, stage) :
    entire_ds = CustomData(self.train_dir, transform = self.transform)
    train_size = int(0.8 * len(entire_ds))
    val_size = len(entire_ds) - train_size
    self.train_ds, self.val_ds = random_split(entire_ds, [train_size, val_size])
    self.test_ds = CustomData(self.test_dir, self.transform)
  def train_dataloader(self) :
    return DataLoader(dataset = self.train_ds,
                              batch_size = self.batch_size,
                              num_workers = self.num_workers,
                              shuffle = True)
  def val_dataloader(self) :
    return DataLoader(dataset = self.val_ds,
                              batch_size = self.batch_size,
                              num_workers = self.num_workers,
                              shuffle = True)
  def test_dataloader(self) :
    return DataLoader(dataset = self.test_ds,
                              batch_size = self.batch_size,
                              num_workers = self.num_workers,
                              shuffle = False)