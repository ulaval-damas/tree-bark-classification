from torch.utils.trainer.plugins.plugin import Plugin
import matplotlib.pyplot as plt
import math
import os


class HistoryPlugin(Plugin):

    def __init__(self, parameters):
        super(HistoryPlugin, self).__init__([(1, 'epoch')])
        self.history = {}
        self.trainer = None

        for parameter in parameters:
            self.history[parameter] = []

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, *args):
        for parameter in self.history.keys():
            self.history[parameter].append(self.trainer.stats[parameter]['epoch_mean'])

    def save_merged_graph(self, file_name, log_path):
        plt.clf()
        self._create_merged_graph()
        plt.savefig(os.path.join(log_path, file_name, file_name + '.png'))

    def show_merged_graph(self):
        plt.clf()
        self._create_merged_graph()
        plt.show()

    def save_data_to_file(self, model_name, log_path):
        file = open(os.path.join(log_path, model_name, 'train_history'), 'w')
        for parameter in self.history.keys():
            file.write('--------------------\n')
            file.write(parameter + '\n')
            file.write('--------------------\n')
            for data in self.history[parameter]:
                file.write(str(data) + '\n')
        file.close()

    def _create_merged_graph(self):
        i = 1
        parameters = self.history.keys()
        n_row = math.ceil(len(parameters) / 2)
        for parameter in self.history.keys():
            plt.subplot(n_row, 2, i)
            self._create_graph(parameter)
            i += 1

        plt.tight_layout()

    def _create_graph(self, parameter):
        plt.xlabel('Epoch')
        plt.ylabel(parameter)
        x = [x for x in range(len(self.history[parameter]))]
        y = self.history[parameter]
        plt.plot(x, y)

    def show_graph(self, parameter):
        plt.clf()
        self._create_graph(parameter)
        plt.show()
