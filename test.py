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
    encoder.load_state_dict(torch.load("../encoder.pth"))

    model = net.Model(encoder, num_classes=10)
    model.load_state_dict(torch.load("../classification_head.pth"))
    model.eval()

    #Select random image from test set
    random_idx = random.randint(0, len(test_set) - 1)
    image, label = test_set[random_idx]

    with torch.nograd():
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

if __name__ == '__main__':
    singleImageTest()