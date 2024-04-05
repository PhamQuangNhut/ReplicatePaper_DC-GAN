from torch import nn
import torch
import pytorch_lightning as pl
from utils import log_tensor_images, show_tensor_images, save_tensor_images
import config
import wandb
class Generator(nn.Module) :
  def __init__(self, z_dim) :
    super().__init__()
    self.z_dim = z_dim
    self.lay1 = nn.Sequential(nn.ConvTranspose2d(in_channels=z_dim,
                                       out_channels=1024,
                                       kernel_size=4,
                                       stride=1,
                                       padding=0),
                              nn.BatchNorm2d(num_features=1024),
                              nn.ReLU()
                              )
    self.lay2 = nn.Sequential(nn.ConvTranspose2d(in_channels=1024,
                                       out_channels=512,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                              nn.BatchNorm2d(num_features=512),
                              nn.ReLU()
                              )
    self.lay3 = nn.Sequential(nn.ConvTranspose2d(in_channels=512,
                                       out_channels=256,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                              nn.BatchNorm2d(num_features=256),
                              nn.ReLU()
                              )
    self.lay4 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                              nn.BatchNorm2d(num_features=128),
                              nn.ReLU()
                              )
    self.lay5 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                       out_channels=3,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                              nn.BatchNorm2d(num_features=3),
                              nn.Tanh()
                              )
  def getNoise(self, cur_batch_size) :
    return torch.randn(cur_batch_size, self.z_dim, 1, 1, device='cuda')
  def forward(self, noise) :
    out = self.lay1(noise)
    out = self.lay2(out)
    out = self.lay3(out)
    out = self.lay4(out)
    out = self.lay5(out)
    return out

class Discriminator(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, stride=1),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 4, stride=2),
            self.make_disc_block(hidden_dim * 4, 1, final_layer=True),
        )

    def make_disc_block(self,
                        input_channels,
                        output_channels,
                        kernel_size=4,
                        stride=2,
                        final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size,
                          stride), nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2))
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size,
                          stride))

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
   
   
# criterion = nn.BCEWithLogitsLoss()
class GAN(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 in_channels=3,
                 hidden_dim=32,
                 z_dim=100,
                 **kwargs):
        super(GAN, self).__init__()

        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()

        self.gen = Generator(z_dim=z_dim)
        self.disc = Discriminator(im_chan=in_channels,
                                  hidden_dim=hidden_dim)
        self.automatic_optimization = False

    def forward(self, noise):
        # in lightning, forward defines the prediction/inference actions
        return self.gen(noise)

    def disc_step(self, x, noise):
        """
        x: real image
        """
        fake_images = self.gen(noise)
        # get discriminator outputs
        real_logits = self.disc(x)
        fake_logits = self.disc(fake_images.detach())
        assert real_logits.shape == fake_logits.shape, f"Real and fake logit shape are different: {real_logits.shape} and {fake_logits.shape}"

        # real loss
        real_loss = self.criterion(real_logits, torch.ones_like(real_logits))
        # fake loss
        fake_loss = self.criterion(fake_logits, torch.zeros_like(fake_logits))
        disc_loss = (fake_loss + real_loss) / 2

        assert disc_loss is not None
        return disc_loss

    def gen_step(self, x, noise):
        # generate fake images
        fake_images = self.gen(noise)

        fake_logits = self.disc(fake_images)
        fake_loss = self.criterion(fake_logits, torch.ones_like(fake_logits))
        gen_loss = fake_loss

        assert gen_loss is not None
        return gen_loss

    def training_step(self, batch, batch_idx):
        x = batch
        x = real = x
        noise = self.gen.getNoise(real.shape[0])

        assert real.shape[1:] == (
            3, 64, 64), f"batch image data shape is incorrect: {real.shape}"

        if batch_idx % config.DISPLAY_STEP == 0:
            fake_images = self.forward(noise)

          #   show_tensor_images(fake_images)
            log_tensor_images(fake_images)
            # show_tensor_images(real)
        lossG = self.gen_step(real, noise)
        lossD = self.disc_step(real, noise)
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        self.manual_backward(lossG)
        opt1.step()
        opt2.zero_grad()
        self.manual_backward(lossD)
        opt2.step()
        wandb.log({"g_loss": lossG, "d_loss": lossD})
        # return lossG, lossD

    def configure_optimizers(self):
        
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.gen.parameters(),
                                 lr=lr,
                                 betas=(config.BETA_1, config.BETA_2))
        opt_d = torch.optim.Adam(self.disc.parameters(),
                                 lr=lr,
                                 betas=(config.BETA_1, config.BETA_2))
        return [opt_g, opt_d]
