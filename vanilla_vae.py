# Original paper: https://arxiv.org/pdf/1312.6114.pdf

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import get_img
from PIL import Image
from torchvision.utils import save_image
from torchvision.utils import make_grid
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    def __init__(self, latent_size=20):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)
        self.fc3 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train():
    run = wandb.init(
        # Set the project where this run will be logged
        project="vanilla_vae_mnist")

    model = VAE(latent_size=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss function for VAE
    def loss_function(recon_x, x, mu, logvar):
        MSE = nn.functional.mse_loss(
            recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.MNIST('./data', transform=transform)
    dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

    # Training loop
    for epoch in range(20):
        for i, (imgs, _) in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            recon_batch, mu, logvar = model(imgs)
            loss = loss_function(recon_batch, imgs, mu, logvar)
            print('Loss: {:.4f}'.format(loss.item()/len(imgs)))
            if i % 100:
                z = torch.randn(6, 20).to(device)
                sample = model.decode(z)
                sample = sample.view(-1, 1, 28, 28)
                grid = make_grid(sample)
                run.log({"image": wandb.Image(Image.fromarray(get_img(grid)))})
            loss.backward()
            run.log({"loss": loss.item()/len(imgs)})
            optimizer.step()

    torch.save(model.state_dict(), './models/vanilla_mnist_vae_final.pth')


def eval():
    model = VAE(latent_size=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load('./models/vanilla_mnist_vae.pth')
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    # generate some images
    with torch.no_grad():
        z = torch.randn(6, 20).to(device)
        sample = model.decode(z)
        sample = sample.view(-1, 1, 28, 28)
        grid = make_grid(sample)
        save_image(grid, './sample.png')


if __name__ == "__main__":
    train()
    # eval()
