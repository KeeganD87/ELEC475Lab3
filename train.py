import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import model as net
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from dataloader import CIFAR100Dataset

def save_loss_plot(losses_train: list, save_path: str):
    #Training loss plot
    plt.plot([i for i in range(len(losses_train))], losses_train, label='Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(save_path)
    plt.close()

#Get device for training (CPU/CUDA)
def get_device(is_cuda: str):
    if(is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def train_transform():
    """
        This transform works by first sizing the image to a slightly larger size than the models expected input.
        Then the model randomly crops a img_sizeximg_size portion of the image, equal to the expected input of the model
        This should help prevent model overfitting
    """
    transform_list = [
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def test_transform():
    #Define testing data transformation
    return transforms.Compose()([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def train(n_epochs: int, optimizer: torch.optim.Optimizer, model: nn.Module, train_loader: DataLoader, device: torch.device, criterion: nn.Module, scheduler):
    print("Training...")
    model.train()

    losses_train = []
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        scheduler.step()

        avg_loss = loss_train / len(train_loader)
        losses_train.append(avg_loss)

        training_str = f"{datetime.datetime.now()}, Epoch: {epoch}, Training Loss: {loss_train/len(train_loader)}"
        print(training_str)

    torch.save(model.state_dict(), args.s)
    return losses_train

def main(args):
    device = get_device(args.cuda)
    print("Device: ", device)

    train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform())
    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, num_workers=2)

    encoder = net.encoder_decoder.encoder #Load model encoder
    encoder.load_state_dict(torch.load(args.l))

    num_classes = 100
    model = net.Model(encoder, num_classes, freeze_weights=False) #Create new model using encoder
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    #Train loop
    losses_train = train(args.e, optimizer, model, train_loader, device, criterion, scheduler)
    save_loss_plot(losses_train, args.p)

    print("Training is complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help='Number of epochs')
    parser.add_argument('-b', type=int, help='Batch Size')
    parser.add_argument('-l', type=str, help='Encoder path')
    parser.add_argument('-s', type=str, help='Model out path')
    parser.add_argument('-p', type=str, help='Loss Plot Path')
    parser.add_argument('-cuda', type=str, default='Y', help="Whether to use CPU or Cuda, use Y or N")
    args = parser.parse_args()
    main(args)
