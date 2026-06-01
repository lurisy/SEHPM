import os
import json
import yaml
from omegaconf import DictConfig, OmegaConf
from torchvision.datasets import CIFAR100



def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]

def get_ordered_class_name(class_order, class_name):
    new_class_name = []
    for i in range(len(class_name)):
        new_class_name.append(class_name[class_order[i]])
    return new_class_name

def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_dataset_class_names( long=False):
    with open("./imagenet100_classes.txt", "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def get_imagenet_r_classes_names(path: object = './datasets/imagenet-r') -> object:
    words_path = os.path.join(path, 'README.txt')
    train_dir = os.path.join(path, 'train')

    wnids = sorted(os.listdir(train_dir))

    wnid_to_words = {}
    with open(words_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            wnid, words = parts
            wnid_to_words[wnid] = words.replace("_", " ")

        class_order =[168, 136, 51, 9, 183, 101, 171, 99, 42, 159, 191, 70, 16, 188, 27, 10, 175, 26, 68, 187, 98, 6, 85, 35, 112, 43, 100, 0, 103, 181, 88, 59, 4, 2, 116, 174, 94, 80, 106, 1, 147, 17, 141, 131, 72, 23, 173, 54, 197, 118, 87, 32, 79, 104, 91, 19, 135, 107, 178, 36, 11, 199, 142, 8, 122, 3, 28, 57, 153, 172, 190, 56, 49, 44, 97, 62, 151, 169, 194, 55, 192, 12, 189, 78, 66, 180, 15, 137, 109, 134, 92, 119, 126, 52, 170, 40, 148, 65, 144, 64, 138, 45, 77, 89, 154, 90, 71, 193, 74, 30, 113, 143, 96, 84, 67, 50, 186, 156, 69, 21, 18, 111, 108, 58, 125, 157, 150, 110, 182, 129, 166, 83, 81, 60, 13, 165, 14, 176, 63, 117, 5, 22, 145, 121, 38, 41, 82, 127, 114, 20, 31, 53, 37, 163, 196, 130, 152, 162, 86, 76, 24, 34, 184, 149, 33, 128, 198, 155, 146, 167, 139, 120, 140, 102, 47, 25, 158, 123, 46, 164, 61, 7, 115, 75, 133, 160, 105, 132, 179, 124, 48, 73, 93, 39, 95, 195, 29, 177, 185, 161]

    if class_order is None:
        class_order = list(range(len(wnids)))

    classes_names = [wnid_to_words[wnids[i]] for i in class_order]
    return classes_names

def get_imagenet_a_classes_names(path: object = './datasets/imagenet-a') -> object:
    words_path = os.path.join(path, 'README.txt')
    train_dir = os.path.join(path, 'train')

    wnids = sorted(os.listdir(train_dir))

    wnid_to_words = {}
    with open(words_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            wnid, words = parts
            wnid_to_words[wnid] = words.replace("_", " ")

        class_order =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
    if class_order is None:
        class_order = list(range(len(wnids)))

    classes_names = [wnid_to_words[wnids[i]] for i in class_order]
    return classes_names

def get_cifar100_classes_names(path: str = './datasets/cifar100'):
    words_path = os.path.join(path, 'README.txt')
    train_dir = os.path.join(path, 'train')

    wnids = sorted(os.listdir(train_dir))

    wnid_to_words = {}
    with open(words_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            wnid, words = parts
            wnid_to_words[wnid] = words.replace("_", " ")

    class_order = [81, 14, 3, 94, 92, 36, 49, 77, 20, 32,
              21, 47, 56, 16, 71, 27, 70, 91, 57, 26,
              74, 90, 79, 97, 23, 73, 87,66, 84, 30,
              24, 88, 19, 93, 29, 65, 35, 4, 86, 58,
              68, 6, 0, 40, 13, 25, 75, 64, 46, 80,
              28, 83, 7, 9, 12, 37, 17, 60, 31, 33,
              8, 2, 63, 10, 18, 55, 95, 78, 85, 62,
              96, 44, 99, 76, 11, 54, 45, 5, 89, 98,
              61, 67, 69, 82, 41, 39, 42, 53, 51, 52,
              50, 1, 34, 22, 38, 15, 43, 59, 72, 48]

    if class_order is None:
        class_order = list(range(len(wnids)))

    classes_names = [wnid_to_words[wnids[i]] for i in class_order]
    return classes_names


def get_tinyimagenet_classes_names(path: object = './datasets/tinyimagenet/tiny-imagenet-200') -> object:

    words_path = os.path.join(path, 'words.txt')
    train_dir = os.path.join(path, 'train')

    wnids = sorted(os.listdir(train_dir))

    wnid_to_words = {}
    with open(words_path, 'r') as f:
        for line in f:
            wnid, words = line.strip().split('\t')
            wnid_to_words[wnid] = words.split(',')[0]

        class_order = [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                        2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192,
                        4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164, 174, 184, 194,
                        6, 16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196,
                        8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198,
                        9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199,
                        7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 147, 157, 167, 177, 187, 197,
                        5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195,
                        3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193,
                        1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]

    if class_order is None:
        class_order = list(range(len(wnids)))

    classes_names = [wnid_to_words[wnids[i]] for i in class_order]
    return classes_names

def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")

def get_workdir(path):
    split_path = path.split("/")
    workdir_idx = split_path.index("clip_based")
    return "/".join(split_path[:workdir_idx+1])


