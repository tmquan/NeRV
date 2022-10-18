import os

from argparse import ArgumentParser
# Pytorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from typing import Dict, List, Optional, Tuple, Type, Union, Callable, Sequence
from pathlib import Path

import numpy as np

import torch
from torch import nn

import torchvision
import torchvision.io
import torchvision.transform

# Monai
from monai.data import Dataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.utils import first, set_determinism, get_seed, MAX_SEED
from monai.transforms import (
    apply_transform, 
    AddChanneld,
    Compose, OneOf, 
    LoadImaged, Spacingd, Lambdad,
    Orientationd, DivisiblePadd, 
    RandFlipd, RandZoomd, RandScaleCropd, CropForegroundd,
    RandAffined,
    Resized, Rotate90d, 
    ScaleIntensityd,
    ScaleIntensityRanged, 
    ToTensord,
)
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

# pytorch 3d
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.structures import Volumes
from pytorch3d.ops.utils import eyes
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
    look_at_rotation,
    look_at_view_transform, 
    get_world_to_view_transform, 
    camera_position_from_spherical_angles,
)

from pytorch3d.renderer import (
    ray_bundle_to_ray_points, 
    RayBundle, 
    VolumeRenderer, 
    VolumeSampler, 
    GridRaysampler, 
    NDCMultinomialRaysampler, NDCGridRaysampler, MonteCarloRaysampler, 
    EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher, 
)

class NeRV3Dataset(Dataset):
    def __init__(self, 
        datadir: str = "/path/to/blender/data/",
        scene: str = "lego",
        shape: int = 800,
        batch_size: int = 32, 
        stage: str = "train"
    ):
        self.datadir = datadir
        self.scene = scene
        self.shape = shape
        self.batch_size = batch_size
        self.config = BlenderDataParserConfig(
            data = Path(os.path.join(self.datadir, self.scene))
        )
        self.dataparser = self.config.setup()
        self.dataparser_outputs = self.dataparser.get_dataparser_outputs(split=stage).as_dict()
        # print(self.dataparser_outputs.keys())
        # print(len(self.dataparser_outputs))
        # print(self.dataparser_outputs)
        # print(len(self.dataparser_outputs["image_filenames"]))
        self.image_filenames = self.dataparser_outputs["image_filenames"]
        self.cameras = self.dataparser_outputs["cameras"]
        self._num_cameras = self.cameras._num_cameras
        self.camera_to_worlds = self.cameras.camera_to_worlds
        self.fx = self.cameras.fx
        self.fy = self.cameras.fy
        self.cx = self.cameras.cx
        self.cy = self.cameras.cy
        self.distortion_params = self.cameras.distortion_params
        self._image_heights = self.cameras._image_heights
        self._image_widths = self.cameras._image_widths
        self.camera_type = self.cameras.camera_type
        # print(vars(self.cameras).keys())
        # print(self.cameras.keys())

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.shape),
                torchvision.transforms.ToTensor()
            ]
        )
        assert len(self.image_filenames) == len(self._num_cameras)

    def __len__(self):
        return len(self.dataparser_outputs["image_filenames"])

    def __getitem__(self, idx):
        imagefile = str(self.image_filenames[idx])
        image = torchvision.io.read_image(imagefile)
        # cam = self.cameras.camera_to_worlds[idx]
        RT = self.cameras.camera_to_worlds[idx]
        R = R[:3, :3]
        T = T[:3, 3]
        fx = self.cameras.fx[idx]
        fy = self.cameras.fy[idx]
        px = self.cameras.cx[idx]
        py = self.cameras.cy[idx]

        # K = [
        #         [fx,   0,   px,   0],
        #         [0,   fy,   py,   0],
        #         [0,    0,    0,   1],
        #         [0,    0,    1,   0],
        # ]
        K = torch.zeros((self._N, 4, 4), dtype=torch.float32)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 2, 0] = px
        K[:, 2, 1] = py
        K[:, 3, 2] = 1.
        K[:, 2, 3] = 1.
        # cam = PerspectiveCameras(R=R, T=T)
        return {"image": image, "R": R, "T": T, "K": K}

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
        # self.config = BlenderDataParserConfig(
        #     data = Path(os.path.join(self.datadir, self.scene))
        # )
        # self.dataparser = self.config.setup()

    def setup(self):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        # self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        # self.train_datasets = InputDataset(self.train_dataparser_outputs)
        # self.train_cameras = self.train_dataparser_outputs.cameras
        self.train_datasets = NeRV3Dataset(
            datadir=self.datadir,
            scene=self.scene,
            shape=self.shape,
            batch_size=self.batch_size, 
            stage="train"
        )
        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        # self.val_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="val")
        # self.val_datasets = InputDataset(self.val_dataparser_outputs)
        self.val_datasets = NeRV3Dataset(
            datadir=self.datadir,
            scene=self.scene,
            shape=self.shape,
            batch_size=self.batch_size, 
            stage="val"
        )
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

    def test_dataloader(self):
        # self.test_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="test")
        # self.test_datasets = InputDataset(self.test_dataparser_outputs)
        self.test_datasets = NeRV3Dataset(
            datadir=self.datadir,
            scene=self.scene,
            shape=self.shape,
            batch_size=self.batch_size, 
            stage="test"
        )
        self.test_loader = DataLoader(
            self.test_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.test_loader

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--shape", type=int, default=800, help="isotropic shape")
#     parser.add_argument("--scene", type=str, default='lego', help="scene name")
#     parser.add_argument("--datadir", type=str, default='/home/qtran/nerfstudio/data/blender/', help="data directory")
#     parser.add_argument("--batch_size", type=int, default=4, help="batch size")

#     hparams = parser.parse_args()

#     datamodule = NeRV3DataModule(
#         datadir = hparams.datadir,
#         scene = hparams.scene, 
#         shape = hparams.shape, 
#         batch_size = hparams.batch_size,
#     )

#     datamodule.setup()
#     print(len(datamodule.train_dataloader()))
#     print(len(datamodule.val_dataloader()))
#     print(len(datamodule.test_dataloader()))
#     debug_data = first(datamodule.train_dataloader())
#     # print(dir(debug_data))
#     # print(vars(debug_data))
#     # print(debug_data.keys())
#     print(debug_data)

class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1):
        super().__init__()
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # densities close to zero.
        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # a neutral gray color everywhere.
        self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))
        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer
        
    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        # Convert the log-space values to the densities/colors
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)
        
        # Instantiate the Volumes object, making sure
        # the densities and colors are correctly
        # expanded batch_size-times.
        volumes = Volumes(
            densities = densities[None].expand(
                batch_size, *self.log_densities.shape),
            features = colors[None].expand(
                batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size,
        )
        
        # Given cameras and volumes, run the renderer
        # and return only the first output value 
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        return self._renderer(cameras=cameras, volumes=volumes)[0]

class NeRV3LightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.shape = hparams.shape
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        self.logsdir = hparams.logsdir
        
        # render_size describes the size of both sides of the 
        # rendered images in pixels. We set this to the same size
        # as the target images. I.e. we render at the same
        # size as the ground truth images.
        # render_size = target_images.shape[1]

        # Our rendered scene is centered around (0,0,0) 
        # and is enclosed inside a bounding box
        # whose side is roughly equal to 3.0 (world units).
        volume_extent_world = 3.0

        raysampler = NDCMultinomialRaysampler( 
            image_width = self.shape,
            image_height = self.shape,
            n_pts_per_ray = 400, #self.shape,
            min_depth = 0.1,
            max_depth = volume_extent_world,
        )

        # 2) Instantiate the raymarcher.
        # Here, we use the standard EmissionAbsorptionRaymarcher 
        # which marches along each ray in order to render
        # each ray into a single 3D color vector 
        # and an opacity scalar.
        raymarcher = EmissionAbsorptionRaymarcher()

        # Finally, instantiate the volumetric render
        # with the raysampler and raymarcher objects.
        visualizer = VolumeRenderer(
            raysampler=raysampler, 
            raymarcher=raymarcher,
        )

        # Instantiate the volumetric model.
        # We use a cubical volume with the size of 
        # one side = 128. The size of each voxel of the volume 
        # is set to volume_extent_world / volume_size s.t. the
        # volume represents the space enclosed in a 3D bounding box
        # centered at (0, 0, 0) with the size of each side equal to 3.
        volume_size = self.shape
        self.volume_model = VolumeModel(
            visualizer,
            volume_size=[volume_size] * 3, 
            voxel_size = volume_extent_world / volume_size,
        )

        self.loss = nn.SmoothL1Loss(reduction="mean", beta=0.1)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=self.lr / 10
        )
        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='common'): 
        images = batch["image"]
        Rs = batch["R"]
        Ts = batch["T"]
        Ks = batch["K"]
        cameras = PerspectiveCameras(R=Rs, T=Ts, K=Ks, device=images.device)
        screens, masks = self.volume_model.forward(cameras=cameras)
        if batch_idx == 0:
            viz2d = torch.cat([images, screens], dim=-1)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        loss = self.loss(images, screens)
        info = {f'loss': loss}

        return info

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self._common_step(batch, batch_idx, optimizer_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str]='common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=int, default=800, help="isotropic shape")
    parser.add_argument("--scene", type=str, default='lego', help="scene name")
    parser.add_argument("--datadir", type=str, default='/home/qtran/nerfstudio/data/blender/', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    hparams = parser.parse_args()

    datamodule = NeRV3DataModule(
        datadir = hparams.datadir,
        scene = hparams.scene, 
        shape = hparams.shape, 
        batch_size = hparams.batch_size,
    )

    datamodule.setup()

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5, 
    )
    lr_callback = LearningRateMonitor(logging_interval='step')
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams, 
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
        ],
        # accumulate_grad_batches=4, 
        strategy="fsdp", #"fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,  #if hparams.use_amp else 32,
    )

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = NeRV3LightningModule(
        hparams = hparams
    )
    model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model


    trainer.fit(
        model, 
        datamodule,
    )

    # test

    # serve