import json
import os
import math
from PIL import Image
import random

class GenerateDataset:

    IGNORED = ['ERB', 'PEG', 'PID']

    def __init__(self, path):
        self.path = path
        self.trees = {}
        self.dataset = {}
        self.classes = []
        self.train = {}
        self.test = {}

    def load_dataset(self, existing_dataset, split=0.5):
        for class_name in os.listdir(self.path):
            if class_name in self.IGNORED:
                continue
            self.classes.append(class_name)

        if existing_dataset != 'None':
            dataset_file = os.path.join(existing_dataset)
            dataset_file = open(dataset_file)
            loaded_dataset = json.load(dataset_file)
            dataset_file.close()
            self.train = loaded_dataset['train']
            self.test = loaded_dataset['test']
        else:
            self.load_new_dataset(split)

    def add_ignore(self):
        files = []
        labels = []
        for class_name in os.listdir(self.path):
            if class_name not in self.IGNORED:
                continue
            self.classes.append(class_name)
            for file in os.listdir(os.path.join(self.path, class_name)):

                file_path = os.path.join(self.path, class_name, file)
                if not file.endswith('NAME.jpg'):
                    img = Image.open(file_path)
                    width, height = img.size
                    if width >= 224 and height >= 224:
                        files.append(file_path)
                        dbh = float(file.split('/')[-1].split('_')[2]) / math.pi
                        labels.append((class_name, dbh))

        self.classes.sort()

        for i, file in enumerate(files):
            self.train['files'].append(file)
            self.train['labels'].append((self.classes.index(labels[i][0]), labels[i][1]))


    def load_new_dataset(self, split):
        for class_name in os.listdir(self.path):
            if class_name in self.IGNORED:
                continue
            self.dataset[class_name] = []
            for file in os.listdir(os.path.join(self.path, class_name)):

                file_path = os.path.join(self.path, class_name, file)
                tree_number = file.split('_')[0]
                if tree_number not in self.trees.keys():
                    self.trees[tree_number] = []

                if tree_number not in self.dataset[class_name]:
                    self.dataset[class_name].append(tree_number)

                if not file.endswith('NAME.jpg'):
                    img = Image.open(file_path)
                    width, height = img.size
                    if width >= 224 and height >= 224:
                        self.trees[tree_number].append(file_path)

        self.classes.sort()
        dataset = self.get_dataset(split)
        self.train = dataset['train']
        self.test = dataset['test']

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def drop_trees_by_class(self, split):
        trees = self.get_trees_in_train_by_class()
        trees_kept = []
        for class_trees in trees.values():
            number_of_trees = len(class_trees)
            indexes = random.sample(range(0, number_of_trees), int(split * number_of_trees))
            for i in indexes:
                trees_kept.append(class_trees[i])
        return trees_kept

    def all_dataset(self, train_size=1.0, tree_size=1.0):

        train_data = self.train

        if tree_size < 1:
            trees = self.drop_trees_by_class(tree_size)

            train = []
            labels = []

            j = 0
            for file in train_data['files']:
                tree_number = int(file.split('/')[-1].split('_')[0])
                if tree_number in trees:
                    train.append(train_data['files'][j])
                    labels.append(train_data['labels'][j])
                j += 1

            train_data = {
                'files': train,
                'labels': labels
            }

        elif train_size < 1:
            train = []
            labels = []
            data = self.get_train_data_by_class()
            for images in data.values():
                number_of_train_images = len(images)
                indexes = random.sample(range(0, number_of_train_images), int(train_size * number_of_train_images))
                for i in indexes:
                    j = images[i]
                    train.append(train_data['files'][j])
                    labels.append(train_data['labels'][j])

            train_data = {
                'files': train,
                'labels': labels
            }

        return {
            'train': train_data,
            'test': self.test
        }

    def get_trees_in_train(self):
        trees = []
        for file in self.train['files']:
            tree_number = int(file.split('/')[-1].split('_')[0])
            if tree_number not in trees:
                trees.append(tree_number)

        return trees

    def get_train_data_by_class(self):
        images_by_class = {x: [] for x in self.classes}
        i = 0
        for file in self.train['files']:
            class_name = file.split('/')[-1].split('_')[1]
            images_by_class[class_name].append(i)
            i += 1

        return images_by_class

    def get_trees_in_train_by_class(self):
        trees_by_class = {x: [] for x in self.classes}
        for file in self.train['files']:
            tree_number = int(file.split('/')[-1].split('_')[0])
            class_name = file.split('/')[-1].split('_')[1]
            if tree_number not in trees_by_class[class_name]:
                trees_by_class[class_name].append(tree_number)

        return trees_by_class

    def get_dataset(self, split):
        train, test = self.get_dataset_trees(split)

        train_files = self.get_files(train)
        train_labels = self.get_labels(train_files)
        test_files = self.get_files(test)
        test_labels = self.get_labels(test_files)

        return {
            'train': {
                'files': train_files,
                'labels': train_labels,
            },
            'test': {
                'files': test_files,
                'labels': test_labels
            }}

    def get_dataset_trees(self, split=1.):
        train = []
        test = []

        for class_name in self.classes:
            trees = self.dataset[class_name]
            random.shuffle(trees)
            split_point = int(split * len(trees))
            train.extend(trees[:split_point])
            test.extend(trees[split_point:])

        return train, test

    def get_files(self, tree_list):
        files = []
        for tree_number in tree_list:
            files.extend(self.trees[tree_number])
        return files

    def get_labels(self, files):
        labels = []
        for file in files:
            label = file.split('/')[-2]
            dbh = float(file.split('/')[-1].split('_')[2]) / math.pi
            labels.append((self.classes.index(label), dbh))
        return labels

    def _generate_k_fold_dataset(self, k):
        folds = [[] for _ in range(k)]
        for class_name in self.classes:
            trees = self.dataset[class_name]
            trees.sort(key=lambda x: int(self.trees[x][0].split('/')[-1].split('_')[2]))
            for i, tree in enumerate(trees):
                folds[i % k].append(tree)

        labels = [[] for _ in range(k)]
        files = [[] for _ in range(k)]
        for i, fold in enumerate(folds):
            files[i].extend(self.get_files(fold))
            labels[i].extend(self.get_labels(files[i]))

        return files, labels

    def get_k_fold_dataset(self, k):
        files, labels = self. _generate_k_fold_dataset(k)

        dataset = {}

        for i in range(k):
            train_files = []
            train_labels = []
            for j in range(k):
                if j != i:
                    train_files.extend(files[j])
                    train_labels.extend(labels[j])
            dataset[i] = {
                'train': {
                    'files': train_files,
                    'labels': train_labels
                },
                'test': {
                    'files': files[i],
                    'labels': labels[i]
                }
            }

        return dataset
