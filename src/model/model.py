import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, resnet_type, pretrained, n_classes):
        super(Model, self).__init__()

        if resnet_type == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(512, n_classes)

        elif resnet_type == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(512, n_classes)

        elif resnet_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(512 * 4, n_classes)

        elif resnet_type == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(512 * 4, n_classes)

        elif resnet_type == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            self.model.classifier = nn.Linear(1024, n_classes)

        self.model.cuda()

    def forward(self, input):
        return self.model(input)

    def set_dilated_convolution(self):
        layers = [self.model.layer1,
                  self.model.layer2,
                  self.model.layer3,
                  self.model.layer4]
        for layer in layers:
            for bottleneck in layer:
                bottleneck.conv1.dilation = (2, 2)
                bottleneck.conv1.padding = (1, 1)

                bottleneck.conv2.dilation = (2, 2)

                bottleneck.conv3.dilation = (2, 2)