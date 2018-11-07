import torch
import torch.nn as nn

from trainer.Trainer import Trainer
from model.model import Model
from torch.autograd import Variable
from dataset.data_loader import get_fp_loader


class FPTrainer(Trainer):
    def __init__(self, batch_size=32, lr=0.01, epoch_lr=None, lr_decay=0., weight_decay=0., n_classes=23, pretrained=True):
        super().__init__(batch_size, lr, epoch_lr, lr_decay, weight_decay, n_classes, pretrained)

    def create_model(self, model_name='resnet50'):
        self.criterion = nn.CrossEntropyLoss()

        self.net = Model(model_name, pretrained=self.pretrained, n_classes=self.dims[0])
        self.net.cuda()

        self.freeze_layers(1)

        self.optimizer = torch.optim.Adam(self._get_parameters(),
                                          self.start_lr, weight_decay=self.weight_decay)

    def freeze_layers(self, n):
        first_params = [self.net.model.conv1.parameters(), self.net.model.bn1.parameters()]
        layers = [self.net.model.layer1.parameters(), self.net.model.layer2.parameters(),
                  self.net.model.layer3.parameters(), self.net.model.layer4.parameters()]
        if n >= 1:
            for params in first_params:
                for param in params:
                    param.requires_grad = False

        for i in range(n - 1):
            layer = layers[i]
            for param in layer:
                param.requires_grad = False

    def get_loader(self, folder):
        loader, _ = get_fp_loader(folder['train']['files'], folder['train']['labels'], self.batch_size)
        return loader

    def create_mini_batch(self, batch_loader):
        batch = next(batch_loader)[1]
        return Variable(batch[0]).cuda(), Variable(batch[1].type(torch.LongTensor).cuda())

    def get_class_predictions(self, output, targets):
        return output.max(1)[1].type_as(targets)

    def update_iteration_info(self, batch_input, output, targets, epoch_acc, loader, loss, j, print_info=True):
        batch_size = batch_input.size(0)
        predictions = self.get_class_predictions(output, targets)
        correct = predictions.eq(targets)
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.sum()
        acc = 100. * correct.data[0] / batch_size
        epoch_acc.append(acc)
        if print_info:
            print('\r', end='')
            print('{} / {} - {:.4f} - {:.2f}%'.format(j + 1, len(loader), loss.data[0], acc), end='',
                  flush=True)

        return epoch_acc