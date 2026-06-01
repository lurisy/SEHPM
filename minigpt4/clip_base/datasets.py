import os
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100,TinyImageNet200, Core50)
from .utils import get_tinyimagenet_classes_names,get_dataset_class_names,get_cifar100_classes_names,get_imagenet_r_classes_names,get_imagenet_a_classes_names
from torchvision import transforms
from continuum.datasets import ImageFolderDataset

class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ]

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class ImageNet_R(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ]


    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "test")
        return super().get_data()

def get_dataset(cfg, is_train, transforms=None):
    dataset, classes_names = None, None
    if cfg.dataset == "cifar100":
        data_path = './datasets/cifar100'
        dataset = CIFAR100(
            data_path=data_path,
            download=True,
            train=is_train,
            # transforms=transforms
        )
        classes_names = get_cifar100_classes_names('./datasets/cifar100')

    elif cfg.dataset == "tinyimagenet":
        data_path = './datasets/tinyimagenet/tiny-imagenet-200'
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_tinyimagenet_classes_names('./datasets/tinyimagenet/tiny-imagenet-200/')

    elif cfg.dataset == "imagenet-r":
        data_path = './datasets/imagenet-r'
        dataset = ImageNet_R(
            data_path,
            train=is_train
        )
        classes_names = get_imagenet_r_classes_names(path='./datasets/imagenet-r')

    elif cfg.dataset == "imagenet-a":
        data_path = './datasets/imagenet-a'
        dataset = ImageNet_A(
            data_path,
            train=is_train
        )
        classes_names = get_imagenet_a_classes_names(path='./datasets/imagenet-a')


    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]

    else:
        raise ValueError(f"'{cfg.dataset}' is an invalid dataset. Please choose a valid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, cl_transforms=None) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    if cl_transforms is None:
        transformation_list = dataset.transformations
    elif hasattr(cl_transforms, "transforms"):
        transformation_list = cl_transforms.transforms
    else:
        transformation_list = cl_transforms

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transformation_list,
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transformation_list,
        )

    elif cfg.scenario == "task-agnostic":
        raise NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        raise ValueError(
            f"You have entered `{cfg.scenario}` which is not a defined scenario, "
            "please choose from {'class', 'domain', 'task-agnostic'}."
        )

    return scenario, classes_names