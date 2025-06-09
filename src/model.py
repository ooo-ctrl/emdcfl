import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import ResNet, BasicBlock

def load_model(model_type, label_len, model_path=None):

    if model_type == 'CNN':
        model = CNN(label_len, 1)
    elif model_type == 'CCNN':
        model = CNN(label_len, 3)
    elif model_type == 'resnet18':
        model = ResNet18(label_len)
    elif model_type == 'resnet50':
        model = ResNet50(label_len)
    elif model_type == 'vitb16':
        model = ViTB16(label_len)
    elif model_type == 'efficientnetb3':
        model = EfficientNetB3(label_len)
    elif model_type == 'convnext':
        model = ConvNeXt(label_len)
    if model_path is not None:
        print('loading from state dict...')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    return model


class CNN(nn.Module):
    def __init__(self, label_len, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv2d(in_channels, 64, 3, padding=1),
                                nn.ReLU()
        )

        self.conv2 = nn.Sequential(
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Sequential(
                                nn.Linear(14 * 14 * 128, 128),
                                nn.ReLU()
        )
        self.fc2 = nn.Linear(128, label_len)
        self.feature_extraction = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.feature_extraction:
            return x
        x = self.fc2(x)
        return x
    

class ResNet18(ResNet):
    def __init__(self, label_len):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        weights = models.ResNet18_Weights.DEFAULT 
        self.load_state_dict(models.resnet18(weights=weights).state_dict())
        num_ftrs = self.fc.in_features    
        self.fc = nn.Linear(num_ftrs, label_len)
        self.feature_extraction = False
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.feature_extraction:
            return x
        x = self.fc(x)

        return x
    

class ResNet50(nn.Module):
    def __init__(self, label_len):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        num_ftrs = resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, label_len)
        self.feature_extraction = False

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.feature_extraction:
            return x
        x = self.fc(x)
        return x


class ViTB16(nn.Module):
    def __init__(self, label_len):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.hidden_dim = self.vit.hidden_dim
        self.fc = nn.Linear(self.hidden_dim, label_len)
        self.feature_extraction = False

    def forward(self, x):
        # This creates tokens (already includes the [CLS] token)
        tokens = self.vit._process_input(x)   # (B, 197, hidden_dim) for 224x224

        n = tokens.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        tokens = torch.cat([batch_class_token, tokens], dim=1)
        encoded = self.vit.encoder(tokens)    # (B, 197, hidden_dim)
        cls_token = encoded[:, 0, :]

        if self.feature_extraction:
            return cls_token

        return self.fc(cls_token)

    
class EfficientNetB3(nn.Module):
    def __init__(self, label_len):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT
        efficientnet = models.efficientnet_b3(weights=weights)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        num_ftrs = efficientnet.classifier[1].in_features
        self.fc = nn.Linear(num_ftrs, label_len)
        self.feature_extraction = False

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.feature_extraction:
            return x
        x = self.fc(x)
        return x

    
class ConvNeXt(nn.Module):
    def __init__(self, label_len):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        convnext = models.convnext_base(weights=weights)
        self.features = convnext.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = convnext.classifier[2].in_features
        self.fc = nn.Linear(num_ftrs, label_len)
        self.feature_extraction = False

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.feature_extraction:
            return x
        x = self.fc(x)
        return x
