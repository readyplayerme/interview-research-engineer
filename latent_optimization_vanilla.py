import torch
from pathlib import Path
from train_classifier import Net
from vanilla_vae import VAE
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from utils import get_img
from PIL import Image
import imageio
import wandb

run = wandb.init(
    # Set the project where this run will be logged
    project="test_vae_guiding_vanilla_vae")

# check if cuda is available below:
device = "cuda" if torch.cuda.is_available() else "cpu"

net = Net()
net.load_state_dict(torch.load('./mnist_net.pth'))
net.to(device)
# Load Vae
vae = VAE()
vae.load_state_dict(torch.load('./vanilla_mnist_vae1.pth'))
vae.to(device)


iterations = 1000
batch_size = 4 # how many images we want to generate at once
number_for_guidance = 3 # the number we want to generate
z = torch.randn(batch_size, vae.latent_size).to(device)

optimizer = torch.optim.SGD([z], lr=0.01)
img_dir = Path("latent_optim_images_vanilla")
img_dir.mkdir(parents=True, exist_ok=True)
normalize_image = transforms.Normalize((0.5,), (0.5,))

for i in range(iterations):
    x_gen = vae.decode(z)
    image = x_gen.view(-1, 1, 28, 28)
    outputs = net(normalize_image(image))
    """
    Write here the code for VAE guidance;
    Log losses and image with wandb;
    Save images in img_dir to produce gif;
    You might need to modify the code outside of this cycle too:)
    """
    grid = make_grid(image, nrow=batch_size)
    grid_pillow = Image.fromarray(get_img(grid))
    if i % 10 == 0:
        print(f"Iteration {i} Loss {loss.item()}")
        grid_pillow.save(f"{img_dir}/latent_optim_{i}.png")
        run.log({"image": wandb.Image(grid_pillow)})


gif_name = 'latent_optim.gif'
file_names = img_dir.glob('*.png')
imageio.mimsave(gif_name, [imageio.imread(str(fn))
                for fn in file_names], fps=5)
run.log({"latent_optim": wandb.Video(gif_name, fps=5)})
