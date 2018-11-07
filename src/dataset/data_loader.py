import torch
import torchvision.transforms as transforms

from dataset.ListDatasetFolder import ListDatasetFolder


def get_loader(files, labels, batch_size):
    transforms_list = [
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    transform = transforms.Compose(transforms_list)
    folder = ListDatasetFolder(files, labels, transform)

    loader = torch.utils.data.DataLoader(
        folder, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    return loader, folder

