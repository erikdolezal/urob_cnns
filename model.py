import torch
import torch.nn as nn
import numpy as np

class ENConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ENConvBlock, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fwd(x)

class DEConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DEConvBlock, self).__init__()
        self.fwd = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fwd(x)

class MyModel(nn.Module):
    """
    MyModel is a custom neural network model for image classification and segmentation and vectorization.
    """

    def __init__(self, output_size=30):
        """Initialize the model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layers.
            output_size (int): Size of the output layer (number of classes).
        """
        super(MyModel, self).__init__()

        self.backbone = nn.ModuleList([
            ENConvBlock(3, 16, 32),    # 64x64 -> 32x32
            ENConvBlock(32, 64, 64),   # 32x32 -> 16x16
            ENConvBlock(64, 128, 128), # 16x16 -> 8x8
            ENConvBlock(128, 256, 256) # 8x8 -> 4x4
        ])

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 16x16 -> 1x1
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        )
        
        self.segmentation_head = nn.ModuleList([
            DEConvBlock(256, 128, 128),  # 4x4 -> 8x8
            DEConvBlock(256, 64, 64),    # 8x8 -> 16x16
            DEConvBlock(128, 32, 32),   # 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            )
        ])   

        

    def forward(self, x):
        """Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (species, masks, embeddings)
            species (torch.Tensor): The species classification logits.
            masks (torch.Tensor): The segmentation masks.
            embeddings (torch.Tensor): The feature embeddings.
        """
        # ‼️‼️‼️‼️ Define the forward pass ‼️‼️‼️‼️

        b1 = self.backbone[0](x)
        b2 = self.backbone[1](b1)
        b3 = self.backbone[2](b2)
        backbone_features = self.backbone[3](b3)

        species = self.classification_head(backbone_features)

        s1 = self.segmentation_head[0](backbone_features)
        s2 = self.segmentation_head[1](torch.cat([s1, b3], dim=1))
        s3 = self.segmentation_head[2](torch.cat([s2, b2], dim=1))
        masks = self.segmentation_head[3](torch.cat([s3, b1], dim=1))

        embeddings = backbone_features.view(backbone_features.size(0), -1)

        return species, masks, embeddings

    def get_embedding(self, x):
        """Get the feature embeddings from the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The feature embeddings.
        """
        # ‼️‼️‼️‼️ Implement the embedding extraction ‼️‼️‼️‼️
        b1 = self.backbone[0](x)
        b2 = self.backbone[1](b1)
        b3 = self.backbone[2](b2)
        backbone_features = self.backbone[3](b3)
        embeddings = backbone_features.view(backbone_features.size(0), -1)  
        return embeddings
