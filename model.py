import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder_decoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
        nn.ReLU() #ReLU for FC layer
    )

class Model(nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder=None, num_classes=10, freeze_weights=True):
        super(Model, self).__init__()

        #Extract the first 32 layers from the encoder for feature extraction
        encoder_list = list(encoder.children())
        self.encoder = nn.Sequential(*encoder_list[:31])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.batch_norm_enc = nn.BatchNorm2d(512)
        self.batch_norm_fc1 = nn.BatchNorm1d(25088)
        self.batch_norm_fc2 = nn.BatchNorm1d(10)

        self.classifier = nn.Sequential(
            nn.Flatten(),   #Flatten input
            nn.Linear(in_features=25088, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        if freeze_weights:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def init_decoder_weights(self, mean, std):
        #Initialize decoder weights with a normal distribution
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def encode(self, x) -> torch.Tensor:
        #Forward pass through encoder
        return self.encoder(x)

    def decode(self, x) -> torch.Tensor:
        #Forward pass through classifier
        return self.classifier(x)

    def forward(self, x) -> torch.Tensor:
        #Forward pass through model
        x = self.encode(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.batch_norm_fc1(x)
        x = self.decode(x)
        x = self.batch_norm_fc2(x)

        x = F.softmax(x, dim=1)
        return x
