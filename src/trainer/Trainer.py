import json
import os

import torch
from torch.backends import cudnn

from utils.HistoryPlugin import HistoryPlugin
from utils.img_utils import copyfile
from utils.utils import make_dir

from time import time


class Trainer:
    def __init__(self, batch_size, lr, epoch_lr, lr_decay, weight_decay, n_classes, pretrained):
        cudnn.benchmark = True

        self.dims = [n_classes, 1]
        self.start_lr = lr
        self.epoch_lr = epoch_lr if epoch_lr else []
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.history = HistoryPlugin(['accuracy', 'loss', 'lr'])
        self.pretrained = pretrained

        self.net = None
        self.criterion = None
        self.optimizer = None

    def train(self, n_epoch, folder, model_name, print_info=True):
        loader = self.get_loader(folder)

        self.create_model(model_name)

        current_lr = self.start_lr
        for i in range(n_epoch):
            start_time = time()
            data_loader = enumerate(loader)
            epoch_loss = []
            epoch_acc = []
            for j in range(len(loader)):
                batch_input, targets = self.create_mini_batch(data_loader)

                self.optimizer.zero_grad()
                try:
                    output = self.net(batch_input)
                except ValueError:
                    pass
                loss = self.criterion(output, targets)
                epoch_loss.append(loss.data[0])
                loss.backward()
                self.optimizer.step()

                epoch_acc = self.update_iteration_info(batch_input, output, targets, epoch_acc, loader, loss, j,
                                                       print_info=print_info)

            print('\r', end='')
            end_time = time()
            self.update_epoch_info(i, n_epoch, current_lr, epoch_loss, epoch_acc, end_time - start_time, print_info=print_info)
            current_lr = self.update_lr(i, current_lr)

    def update_iteration_info(self, batch_input, output, targets, epoch_acc, loader, loss, j, print_info=True):
        raise NotImplementedError()

    def update_epoch_info(self, epoch, n_epoch, lr, loss, acc, epoch_time, print_info=True):
        if print_info:
            print('-----------------------------------------------------------')
            print('Epoch {} / {}'.format(epoch + 1, n_epoch))
            print('Lr: {}'.format(lr))
            print('Loss: {:.4f}'.format(sum(loss) / len(loss)))
            print('Accuracy: {:.2f}'.format(sum(acc) / len(acc)))
            print('Time: {:.2f} s'.format(epoch_time))
            print('-------------------------------------------------------------')

        self.history.history['lr'].append(lr)
        self.history.history['loss'].append(sum(loss) / len(loss))
        self.history.history['accuracy'].append(sum(acc) / len(acc))

    def update_lr(self, epoch, old_lr):
        new_lr = old_lr
        if epoch in self.epoch_lr:
            for i, param_group in enumerate(self.optimizer.param_groups):
                new_lr = max(old_lr * self.lr_decay, 0)
                param_group['lr'] = new_lr

        return new_lr

    def save_train_data(self, model_name, log_path, config_path, dataset, save_graph=False):
        make_dir(log_path + '/' + model_name)

        if save_graph:
            self.history.save_merged_graph(model_name, log_path)

        self.history.save_data_to_file(model_name, log_path)
        torch.save(self.net, os.path.join(log_path, model_name, model_name))
        copyfile(config_path, log_path + '/' + model_name + '/{}.log'.format(model_name))
        with open(log_path + '/' + model_name + '/dataset', 'w') as file:
            file.write(json.dumps(dataset))

    def create_model(self, model_name):
        raise NotImplementedError()

    def get_loader(self, folder):
        raise NotImplementedError()

    def create_mini_batch(self, batch_loader):
        raise NotImplementedError()

    def get_class_predictions(self, output, targets):
        raise NotImplementedError()

    def _get_parameters(self):
        for parameter in self.net.parameters():
            if parameter.requires_grad:
                yield parameter