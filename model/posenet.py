import torch.nn as nn
from torchvision.models import mobilenet_v2

class LightweightPoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(LightweightPoseNet, self).__init__()

        # Use MobileNetV2 as backbone
        mobilenet = mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[:-1])

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_keypoints, 1)
        )

    def forward(self, x):
        # Backbone
        features = self.backbone(x)

        # Decoder
        heatmaps = self.decoder(features)

        return heatmaps