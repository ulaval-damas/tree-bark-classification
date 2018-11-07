from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math


class ConfusionMatrix:
    def __init__(self, y_true, y_pred, classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes

        self.cm = confusion_matrix(y_true, y_pred, labels=classes)

    def plot_custom(self, save_path=None):
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 18})
        labels = [
            'BOJ', 'BOP', 'CHR', 'EPB', 'EPN', 'EPO', 'EPR', 'ERR', 'ERS', 'FRA',
            'HEG', 'MEL', 'ORA', 'OSV', 'PET', 'PIB', 'PIR', 'PRU', 'SAB', 'THO'
        ]
        order = ['SAB', 'ERR', 'ERS', 'BOJ', 'BOP', 'HEG', 'FRA', 'MEL', 'OSV', 'EPO', 'EPB', 'EPN', 'EPR', 'PIR',
                 'PIB', 'PET', 'CHR', 'THO', 'PRU', 'ORA']
        association = {
            'SAB': 1,
            'ERR': 3,
            'ERS': 4,
            'BOJ': 5,
            'BOP': 6,
            'HEG': 7,
            'FRA': 8,
            'MEL': 9,
            'OSV': 10,
            'EPO': 11,
            'EPB': 12,
            'EPN': 13,
            'EPR': 14,
            'PIR': 16,
            'PIB': 17,
            'PET': 19,
            'CHR': 20,
            'THO': 21,
            'PRU': 22,
            'ORA': 23
        }

        name = {
            1: '\emph{Abies balsamea} - Balsam fir',
            2: '\emph{Acer platanoides} - Norway maple',
            3: '\emph{Acer rubrum} - Red maple',
            4: '\emph{Acer saccharum} - Sugar maple',
            5: '\emph{Betula alleghaniensis} - Yellow birch',
            6: '\emph{Betula papyrifera} - White birch',
            7: '\emph{Fagus grandifolia} - American beech',
            8: '\emph{Fraxinus americana} - White ash',
            9: '\emph{Larix laricina} - Tamarack',
            10: '\emph{Ostrya virginiana} - American hophornbeam',
            11: '\emph{Picea abies} - Norway spruce',
            12: '\emph{Picea glauca} - White spruce',
            13: '\emph{Picea mariana} - Black spruce',
            14: '\emph{Picea rubens} - Red spruce',
            15: '\emph{Pinus rigida} - Pitch pine',
            16: '\emph{Pinus resinosa} - Red pine',
            17: '\emph{Pinus strobus} - Eastern white pine',
            18: '\emph{Populus grandidentata} - Big-tooth aspen',
            19: '\emph{Populus tremuloides} - Quaking aspen',
            20: '\emph{Quercus rubra} - Northern red oak',
            21: '\emph{Thuja occidentalis} - Northern white cedar',
            22: '\emph{Tsuga canadensis} - Eastern Hemlock',
            23: '\emph{Ulmus americana} - American elm'
        }

        new_cm = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for i, label in enumerate(labels):
            row_col = order.index(label)
            new_cm[row_col, :] = self.cm[i, :]

        for i in range(len(self.classes)):
            new_column = -1
            while new_column != i:
                new_column = np.argmax(new_cm[:, i])
                temp = np.copy(new_cm[:, new_column])
                new_cm[:, new_column] = new_cm[:, i]
                new_cm[:, i] = temp

        self.cm = new_cm

        from copy import deepcopy
        normalized_cm = deepcopy(self.cm)
        for i in range(len(self.classes)):
            total = sum(self.cm[i])
            for j in range(len(self.classes)):
                normalized_cm[i][j] = (self.cm[i][j] / total)*100


        fig, ax = plt.subplots(tight_layout=True)
        #ax.set_title('Confusion matrix\n{:.2f} %'.format(self.get_global_correct() * 100))
        ax.matshow(normalized_cm, cmap=plt.cm.Greys, aspect='auto')

        ax2 = ax.twinx()
        ax2.matshow(normalized_cm, cmap=plt.cm.Greys, aspect='auto', vmin=78, vmax=100)
        ax2.set_yticks([i for i in range(len(self.classes))])
        ax2.set_yticklabels(['{:.2f} \%'.format(self.get_avg_correct_for_class(i) * 100) for i in range(len(self.classes))], fontsize=18)

        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                c = self.cm[i, j]
                ax2.text(j, i, str(c), va='center', ha='center', color="white" if normalized_cm[i, j] > 89 else "black")

        ax.set_xlabel('Pred', fontsize=20)
        ax.xaxis.tick_bottom()
        ax.set_xticks([i for i in range(len(self.classes))])
        ax.set_xticklabels(['{}'.format(association[o]) for o in order], fontsize=18)

        ax.set_ylabel('True', fontsize=20)
        ax.set_yticks([i for i in range(len(self.classes))])
        ax.set_yticklabels(['{} - {}'.format(association[o], name[association[o]]) for o in order], fontsize=18)

        plt.show()
        if save_path:
            plt.savefig(save_path, format='png')

    def __str__(self):
        n_digits = int(1 + math.ceil(math.log10(np.max(self.cm))))
        return_str = ''
        for i in range(self.cm.shape[0]):
            return_str += self.classes[i] + ' | '
            for j in range(self.cm.shape[1]):
                return_str += ('{:' + str(n_digits) + 'd} ').format(self.cm[i, j])

            return_str += ' | {:.2f} %\n'.format(self.get_avg_correct_for_class(i) * 100)

        return_str += '      '
        for i in range(len(self.classes)):
            return_str += ('{:-<' + str(n_digits+1) + '}').format('')
        return_str += '\n'
        return_str += '      '
        for class_name in self.classes:
            return_str += ('{:>' + str(n_digits) + '} ').format(class_name)

        return_str += ' | {:.2f} %\n'.format(self.get_global_correct() * 100)

        return return_str

    def get_avg_correct_for_class(self, class_index):
        row = self.cm[class_index, :]
        correct = row[class_index]
        return correct / np.sum(row)

    def get_global_correct(self):
        correct = 0
        for i in range(len(self.classes)):
            correct += self.cm[i, i]
        return correct / np.sum(self.cm)

    def normalize_to(self, normalize_factor):
        for i, row in enumerate(self.cm):
            total_row = np.sum(row)
            if total_row != 0:
                for j, data in enumerate(row):
                    normalized_data = data * normalize_factor / total_row
                    self.cm[i][j] = normalized_data

