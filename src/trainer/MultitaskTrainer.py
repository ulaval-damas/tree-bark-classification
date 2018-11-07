import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dataset.data_loader import get_loader
from model.dcnet import DeepCollaborationNetwork, MultiTaskCriterion
from trainer.Trainer import Trainer


class MultitaskTrainer(Trainer):
    def __init__(self, batch_size=32, lr=0.01, epoch_lr=None, lr_decay=0.,
                 weight_decay=0., n_classes=23, pretrained=True):

        super().__init__(batch_size, lr, epoch_lr, lr_decay, weight_decay, n_classes, pretrained)

    def create_model(self, model_name):
        criterions = [nn.CrossEntropyLoss(), nn.L1Loss()]
        weights = [1, 0.10]
        self.criterion = MultiTaskCriterion(criterions, weights)

        self.net = DeepCollaborationNetwork(model_name, self.dims, pretrained=self.pretrained)
        self.net.cuda()

        self.freeze_layers(1)

        self.optimizer = optim.Adam(self._get_parameters(), lr=self.start_lr, weight_decay=self.weight_decay)

    def freeze_layers(self, n):
        for column in self.net.columns:
            for i in range(n):
                for param in column[i].parameters():
                    param.requires_grad = False

    def get_loader(self, folder):
        loader, _ = get_loader(folder['train']['files'], folder['train']['labels'], self.batch_size)
        loader.dataset.set_multitask()
        return loader

    def create_mini_batch(self, batch_loader):
        batch = next(batch_loader)[1]
        final_targets = [Variable(batch[1].type(torch.LongTensor)).cuda(),
                         Variable(batch[2].type(torch.FloatTensor)).cuda()]
        return Variable(batch[0]).cuda(), final_targets

    def get_class_predictions(self, output, targets):
        return output[0].max(1)[1].type_as(targets[0])

    def update_iteration_info(self, batch_input, output, targets, epoch_acc, loader, loss, j, print_info=True):
        batch_size = batch_input.size(0)
        predictions = self.get_class_predictions(output, targets)
        correct = predictions.eq(targets[0])
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