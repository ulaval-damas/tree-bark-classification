from torch.utils.data.dataset import Dataset
from PIL import Image
from random import choice


class ListDatasetFolder(Dataset):

    def __init__(self, files, labels, transforms=None):
        self.dataset = files
        self.labels = labels
        self.transforms = transforms
        self.multitask = False

        self.classes = {}
        self.load_trees(files)

    def load_trees(self, files):

        for i, file in enumerate(files):
            tree_number = int(file.split('/')[-1].split('_')[0])
            class_name = file.split('/')[-1].split('_')[1]
            if class_name not in self.classes.keys():
                self.classes[class_name] = {}

            if tree_number in self.classes[class_name].keys():
                self.classes[class_name][tree_number].append(i)
            else:
                self.classes[class_name][tree_number] = [i]

    def set_multitask(self):
        self.multitask = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        class_name = choice(list(self.classes.keys()))
        tree_number = choice(list(self.classes[class_name].keys()))
        index = choice(self.classes[class_name][tree_number])

        path = self.dataset[index]
        img = Image.open(path)
        if self.transforms:
            img = self.transforms(img)

        if self.multitask:
            item = (img, self.labels[index][0], self.labels[index][1])
        else:
            item = (img, self.labels[index][0])

        return item
