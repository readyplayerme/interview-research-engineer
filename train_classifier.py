import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

wandb.init(project="mnist-classifier")
device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def run_training():
    # Load MNIST dataset
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    # Initialize the network, loss function and optimizer
    net = Net()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.fc1.parameters(), lr=0.1)

    # Training loop
    # Please assign device to tensors, 
    # Log your loss function with wandb
    for epoch in range(5):  # 5 epochs
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.detach()
            loss = criterion(outputs, labels)
            loss.backward()
            if i % 10 == 0:
                print('Epoch: %d, Iteration: %d, Loss: %f' %
                      (epoch, i, loss.item()))
            optimizer.step()

    # Testing the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    torch.save(net.state_dict(), './mnist_net.pth')


if __name__ == "__main__":
    run_training()
