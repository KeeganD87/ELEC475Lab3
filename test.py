import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import model as net
import matplotlib.pyplot as plt
import random

def singleImageTest():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load CIFAR10
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    encoder = net.encoder_decoder.encoder  # Load model encoder
    encoder.load_state_dict(torch.load("encoder.pth"))

    model = net.Model(encoder, num_classes=10)
    model.load_state_dict(torch.load("classification_head.pth"))
    model.eval()

    #Select random image from test set
    random_idx = random.randint(0, len(test_set) - 1)
    image, label = test_set[random_idx]

    with torch.no_grad():
        image = image.unsqueeze(0)
        outputs = model(image)

    _, predicted = torch.topk(outputs, 5, dim=1)
    predicted = predicted[0].tolist()
    probabilities = torch.softmax(outputs, dim=1)[0].tolist()

    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #Display image
    image = image.squeeze(0)
    image = image / 2 + 0.5     #Denormalize image
    image = image.permute(1, 2, 0)

    plt.imshow(image)
    plt.title(f"Actual class: {class_labels[int(label)]}")
    plt.show()

    #Top 5 predicted classes and corresponsing probabilities
    print("Top 5 predicted classes: ")
    for i in range(5):
        class_idx = predicted[i]
        class_probabilities = probabilities[class_idx]
        class_name = class_labels[class_idx]
        print(f"{i+1}. Class: {class_name}, Probability: {class_probabilities:.4f}")

def find_error_rates(args):
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])

    test_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l))

    model = net.Model(encoder, num_classes=args.c)
    model.load_state_dict(torch.load(args.s))
    model.eval()

    if args.cuda.lower() == 'y' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device: ", device)

    model.to(device)
    encoder.to(device)

    #Initialize variables for top 1 and top 5 results
    top1_count = 0
    top5_count = 0

    num_iterations = len(test_loader)
    for i, data in enumerate(test_loader, 0):
        images, labels = data

        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)

        _, predicted = torch.topk(outputs, 5, dim=1)
        predicted = predicted[0].tolist()
        probabilities = torch.softmax(outputs, dim=1)[0].tolist()

        if predicted[0] == labels[0]:
            top1_count += 1
        if labels[0] in predicted:
            top5_count += 1

        if i % 100 == 0:
            print(f"Iteration {i}/{num_iterations}")

    top1_percent = (top1_count / len(test_set)) * 100
    top5_percent = (top5_count / len(test_set)) * 100
    top1_error = 100 - top1_percent
    top5_error = 100 - top5_percent

    print(f"Top 1: {top1_percent}% - Error: {top1_error}%")
    print(f"Top 5: {top5_percent}% - Error: {top5_error}%")

if __name__ == '__main__':
    #singleImageTest()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=10, help='Number of classes')
    parser.add_argument('-l', type=str, default="encoder.pth", help='Encoder path')
    parser.add_argument('-s', type=str, default="classification_head.pth", help='Model path')
    parser.add_argument('-cuda', type=str, default='Y', help="Whether to use CPU or Cuda, use Y or N")
    args = parser.parse_args()
    find_error_rates(args)
