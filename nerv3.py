import os

from argparse import ArgumentParser

from typing import Dict, List, Optional, Tuple, Type, Union, Callable, Sequence
from pathlib import Path

import numpy as np

import torch
from torch import nn

# Monai
from monai.data import Dataset, DataLoader
from monai.data import list_data_collate, decollate_batch


# Nerf Studio
# from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
# from nerfstudio.cameras.rays import RayBundle
# from nerfstudio.configs.base_config import InstantiateConfig
# from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
# from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
# from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig 
# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
# from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
# from nerfstudio.data.utils.datasets import InputDataset
from nerfstudio.configs.base_config import Config
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.utils.datasets import InputDataset

# Pytorch Lightning
from pytorch_lightning import LightningDataModule

class NeRV3DataModule(LightningDataModule):
    def __init__(self, 
        datadir: str = "/path/to/blender/data/",
        scene: str = "lego",
        shape: int = 800,
        batch_size: int = 32
    ):
        self.datadir = datadir
        self.scene = scene
        self.shape = shape
        self.batch_size = batch_size
        self.config = BlenderDataParserConfig(
            data = Path(os.path.join(self.datadir, self.scene))
        )
        self.dataparser = self.config.setup()

    def setup(self):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_datasets = InputDataset(dataparser_outputs)
        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        dataparser_outputs = self.dataparser.get_dataparser_outputs(split="val")
        self.val_datasets = InputDataset(dataparser_outputs)
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

    def test_dataloader(self):
        dataparser_outputs = self.dataparser.get_dataparser_outputs(split="test")
        self.test_datasets = InputDataset(dataparser_outputs)
        self.test_loader = DataLoader(
            self.test_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.test_loader

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=int, default=800, help="isotropic shape")
    parser.add_argument("--scene", type=str, default='lego', help="scene name")
    parser.add_argument("--datadir", type=str, default='/home/qtran/nerfstudio/data/blender/', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()

    datamodule = NeRV3DataModule(
        datadir = hparams.datadir,
        scene = hparams.scene, 
        shape = hparams.shape, 
        batch_size = hparams.batch_size,
    )

    datamodule.setup()
    print(len(datamodule.train_dataloader()))
    print(len(datamodule.val_dataloader()))
    print(len(datamodule.test_dataloader()))

