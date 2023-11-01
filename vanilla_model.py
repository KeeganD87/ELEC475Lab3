import torch
import torch.nn as nn
import torchvision.models as models

#Load the VGG model architecture (you can choose VGG16, VGG19, etc.)
vgg_model = models.vgg16(pretrained=True)

#Load the saved weights and state dictionary from the .pth file
saved_model_state = torch.load('encoder.pth')

#Load the saved weights into the model
vgg_model.load_state_dict(saved_model_state)

#Ensure the model is in evaluation mode if you're using it for feature extraction
vgg_model.eval()

class CustomFrontEnd(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CustomFrontEnd, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x= self.softmax(x)
        return x

class VGGCustomModel(nn.Module):
    def __init__(self, backend, frontend):
        super(VGGCustomModel, self).__init__()
        self.backend = backend
        self.frontend = frontend

    def forward(self, x):
        x = self.backend(x)
        x = x.view(x.size(0), -1)
        x = self.frontend(x)
        return x