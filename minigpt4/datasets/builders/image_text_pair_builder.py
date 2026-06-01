import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
import glob
from braceexpand import braceexpand

@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass



    def build(self,task_id: int = 0):
        self.build_processors()
        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # 展开 {00000..01255}.tar 的路径为真实文件列表
        if "{" in build_info.storage:
            # 用 shell 展开表达式（例如 {00000..01255}）转为列表
            from braceexpand import braceexpand
            all_tars = list(braceexpand(build_info.storage))
        elif isinstance(build_info.storage, str):
            all_tars = [build_info.storage]
        elif isinstance(build_info.storage, list):
            all_tars = build_info.storage
        else:
            raise ValueError("Unsupported storage format.")

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=all_tars,
            task_id=task_id,
        ).inner_dataset

        return datasets
'''
    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
'''

@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass


    def build(self, task_id: int = 0):
        self.build_processors()
        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # 自动展开 {00000..10488}
        if isinstance(build_info.storage, str) and "{" in build_info.storage:
            expanded_paths = list(braceexpand(build_info.storage))
        elif isinstance(build_info.storage, list):
            expanded_paths = build_info.storage
        else:
            expanded_paths = [build_info.storage]

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=expanded_paths,
            task_id=task_id,
        ).inner_dataset

        return datasets


'''
    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
'''

@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self, task_id: int):
        logging.info("Building datasets...")
        self.build_processors()

        task_id = int(task_id)

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        dataset_cls = self.train_dataset_cls
        cap_name = '20_20_order3_2000exp/task' + str(task_id) + '.json'

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, cap_name)],
            vis_root=os.path.join(storage_path, 'image'),
            task_id=task_id,
        )


        return datasets


