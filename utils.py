from torchvision.utils import make_grid
import wandb
import matplotlib.pyplot as plt
from PIL import Image
def show_tensor_images(image_tensor,
                       num_images=25,
                       ret=False):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    if ret:
        return image_grid.permute(1, 2, 0).squeeze()
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
def save_tensor_images(image_tensor, save_path, num_images=25):
    '''
    Function for saving images: Given a tensor of images, number of images, and
    save path, saves the images in a grid format.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    image_grid = image_grid.permute(1, 2, 0).squeeze().numpy() * 255
    image_grid = Image.fromarray(image_grid.astype('uint8'))
    image_grid.save(save_path)
def log_tensor_images(image_tensor, num_images=25, name="Generated Images"):
    '''
    Function for logging images on wandb: Given a tensor of images and number of images,
    logs the images as a grid to wandb.
    '''
    # Normalize the image tensor to [0, 1]
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # Convert image grid to numpy and log to wandb
    wandb.log({name: [wandb.Image(image_grid.permute(1, 2, 0).numpy(), caption="Generated Images")]})
