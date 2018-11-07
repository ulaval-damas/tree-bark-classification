import os
from shutil import rmtree

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def empty_dir(dir_path):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            rmtree(file_path)