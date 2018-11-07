# Adapted from https://github.com/ltrottier/deep-collaboration-network

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskCriterion:
    def __init__(self, criterions, weights):
        self.criterions = criterions
        self.weights = weights
        self.n_criterions = len(self.criterions)

    def __call__(self, predictions, targets):
        self.criterions_loss = []
        self.criterions_weighted_loss = []
        self.loss = 0

        for i in range(self.n_criterions):
            cur_loss = self.criterions[i](predictions[i], targets[i])
            self.criterions_loss.append(cur_loss)
            cur_loss = cur_loss * self.weights[i]
            self.criterions_weighted_loss.append(cur_loss)
            self.loss = self.loss + cur_loss

        return self.loss


def network_as_series_of_blocks(name, pretrained):
    blocks = []

    if name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained)
        n_features = [64, 64, 128, 256, 512]
        blocks.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
        blocks.append(net.layer1)
        blocks.append(net.layer2)
        blocks.append(net.layer3)
        blocks.append(net.layer4)

    elif name == 'resnet34':
        net = torchvision.models.resnet34(pretrained=pretrained)
        n_features = [64, 64, 128, 256, 512]
        blocks.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
        blocks.append(net.layer1)
        blocks.append(net.layer2)
        blocks.append(net.layer3)
        blocks.append(net.layer4)

    elif name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=pretrained)
        n_features = [64, 256, 512, 1024, 2048]
        blocks.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
        blocks.append(net.layer1)
        blocks.append(net.layer2)
        blocks.append(net.layer3)
        blocks.append(net.layer4)

    elif name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=pretrained)
        n_features = [64, 256, 512, 1024, 2048]
        blocks.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
        blocks.append(net.layer1)
        blocks.append(net.layer2)
        blocks.append(net.layer3)
        blocks.append(net.layer4)

    elif name == 'alexnet':
        net = torchvision.models.alexnet(pretrained=pretrained)
        features = net.features
        n_features = [64, 192, 256]

        blocks.append(nn.Sequential(*[features[i] for i in range(0, 3)]))
        blocks.append(nn.Sequential(*[features[i] for i in range(3, 6)]))
        blocks.append(nn.Sequential(*[features[i] for i in range(6, 13)]))

    model = nn.Sequential(*blocks)

    return model, n_features


class CollaborativeBlock(nn.Module):
    def __init__(self, n_inputs, n_features):
        super(CollaborativeBlock, self).__init__()

        self.n_inputs = n_inputs
        self.n_features = n_features

        def gen_central_aggregation(n_in, n_out):
            layer = nn.Sequential(
                nn.Conv2d(n_in, n_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU())
            return layer

        def gen_task_aggregation(n_in, n_out):
            layer = nn.Sequential(
                nn.Conv2d(n_in, n_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_out))
            return layer

        n_in = n_features * n_inputs
        n_out = n_features * n_inputs // 4
        self.central_aggregation = gen_central_aggregation(n_in, n_out)

        n_in = n_features + n_out
        n_out = n_features
        task_aggregation = []
        for i in range(n_inputs):
            task_aggregation.append(gen_task_aggregation(n_in, n_out))
        self.task_aggregation = nn.Sequential(*task_aggregation)

    def forward(self, inputs):
        z = torch.cat(inputs, 1)
        z = self.central_aggregation(z)

        outputs = []
        for i, x in enumerate(inputs):
            y = torch.cat([x, z], 1)
            y = F.relu(x + self.task_aggregation[i](y))
            outputs.append(y)

        return outputs


class DeepCollaborationNetwork(nn.Module):
    def __init__(self, underlying_network_name, out_dims, pretrained):
        super(DeepCollaborationNetwork, self).__init__()

        self.out_dims = out_dims
        self.n_cols = len(out_dims)

        # Create networks
        n_features = network_as_series_of_blocks(underlying_network_name, pretrained)[1]
        self.n_blocks = len(n_features)
        columns = [network_as_series_of_blocks(underlying_network_name, pretrained)[0] for _ in range(self.n_cols)]
        self.columns = nn.Sequential(*columns)

        # Create collaborative blocks
        collaborative_blocks = [CollaborativeBlock(self.n_cols, nf) for nf in n_features]
        self.collaborative_blocks = nn.Sequential(*collaborative_blocks)

        # Create fc layers
        def fc_block(dim_in, dim_out):
            dim_h = (dim_in + dim_out) // 2
            block = nn.Sequential(
                nn.Linear(dim_in, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_out))
            return block
        self.fcs = []
        for out_dim in self.out_dims:
            self.fcs.append(fc_block(n_features[-1], out_dim))
        self.fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        inputs = [x] * self.n_cols
        for i in range(self.n_blocks):
            hiddens = []
            for j, input in enumerate(inputs):
                hiddens.append(self.columns[j][i](input))
            inputs = self.collaborative_blocks[i](hiddens)

        outputs = []
        for i, x in enumerate(inputs):
            z = F.avg_pool2d(x, kernel_size=x.size()[2:])
            z = z.view(z.size(0), -1)
            outputs.append(self.fcs[i](z))

        return outputs