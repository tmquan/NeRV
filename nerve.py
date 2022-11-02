import resource
import torchvision
import torch

from renderer import *
from raysampler import *
from raymarcher import *
from datamodule import NeRVDataModule
from pytorch3d.renderer import (
    ray_bundle_to_ray_points,
    RayBundle,
    ImplicitRenderer,
    VolumeRenderer,
    VolumeSampler,
    GridRaysampler,
    NDCMultinomialRaysampler, NDCGridRaysampler, MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher,
)
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
from pytorch3d.transforms import Transform3d
from pytorch3d.ops.utils import eyes
from pytorch3d.structures import Volumes
from pytorch3d.common.compat import meshgrid_ij

from monai.networks.layers import *  # Reshape
from monai.networks.nets import *  # Unet, DenseNet121, Generator

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Optional
import os
import glob
from re import T
import warnings
warnings.filterwarnings("ignore")


torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.multiprocessing.set_sharing_strategy('file_system')


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


class NeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.logsdir = hparams.logsdir
        self.lr = hparams.lr
        self.shape = hparams.shape
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.save_hyperparameters()

        raysampler = NDCMultinomialRaysampler(  # NDCGridRaysampler(
            image_width=self.shape,
            image_height=self.shape,
            n_pts_per_ray=400,  # self.shape,
            min_depth=0.1,
            max_depth=4.5,
        )

        raymarcher = EmissionAbsorptionRaymarcherFrontToBack()  # X-Ray Raymarcher

        renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        )

        self.visualizer = FigureRenderer(
            renderer=renderer
        )

        # self.unet_model = Unet(
        #     dim=64,
        #     dim_mults=(1, 2, 4, 8),
        #     channels=1,
        # )
        # self.numsteps = 60

        self.loss = nn.SmoothL1Loss(reduction="mean", beta=0.01)

        self.clarity_net = nn.Sequential(
            Unet(
                spatial_dims=2,
                in_channels=1,
                out_channels=self.shape,
                channels=(64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2),
                num_res_units=4,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
            ),
            Reshape(*[1, self.shape, self.shape, self.shape]),
            nn.Tanh()
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
            ),
            nn.Tanh()
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2,
                out_channels=1,
                channels=(64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
            ),
            nn.Tanh()
        )

    def forward(self, figures):
        clarity = self.clarity_net(figures * 2.0 - 1.0)
        density = self.density_net(clarity)
        volumes = self.mixture_net(
            torch.cat([clarity, density], dim=1)) * 0.5 + 0.5
        return volumes

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        # Construct the fixed cameras
        dist0 = 3.0 * torch.ones(self.batch_size, device=_device)
        elev0 = torch.zeros(self.batch_size, device=_device)
        azim0 = torch.zeros(self.batch_size, device=_device)
        R0, T0 = look_at_view_transform(dist=dist0, elev=elev0, azim=azim0)
        cameras0 = FoVPerspectiveCameras(R=R0, T=T0).to(_device)
        figures = self.visualizer.forward(image3d=image3d, cameras=cameras0)

        volumes = self.forward(figures)
        screens = self.visualizer.forward(image3d=volumes, cameras=cameras0)
        reconst = self.forward(image2d)
        picture = self.visualizer.forward(image3d=reconst, cameras=cameras0)

        # input2d = torch.cat([figures, image2d], dim=0)
        # recon3d = self.forward(input2d)
        # volumes, reconst = recon3d[:self.batch_size], recon3d[self.batch_size:]
        # screens = self.visualizer.forward(image3d=volumes, cameras=cameras0)
        # picture = self.visualizer.forward(image3d=reconst, cameras=cameras0)

        if batch_idx == 0:
            viz2d = torch.cat([image3d[..., self.shape//2, :],
                               figures,
                               volumes[..., self.shape//2, :],
                               screens,
                               image2d,
                               reconst[..., self.shape//2, :],
                               picture,
                               ], dim=-2).transpose(2, 3)
            grid = torchvision.utils.make_grid(
                viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(
                0., 1.), self.current_epoch*self.batch_size + batch_idx)

        im3d_loss = self.loss(image3d, volumes)
        im2d_loss = self.loss(image2d, picture) + self.loss(figures, screens)
        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage == 'train'),
                 prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage == 'train'),
                 prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)

        info = {f'loss': im3d_loss + im2d_loss}
        return info

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str] = 'common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False,
                 prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=self.lr / 10
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="NeRV")
    parser.add_argument("--notification_email", type=str,
                        default="quantm88@gmail.com")

    # Model arguments
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256,
                        help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=501,
                        help="number of epochs")
    parser.add_argument("--train_samples", type=int,
                        default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int,
                        default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int,
                        default=400, help="test samples")

    parser.add_argument("--weight_decay", type=float,
                        default=1e-4, help="Weight decay")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="path to checkpoint")
    parser.add_argument("--logsdir", type=str,
                        default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str,
                        default='data', help="data directory")

    parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

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
    tensorboard_logger = TensorBoardLogger(
        save_dir=hparams.logsdir, log_graph=True)

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
        # strategy="ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        strategy="fsdp",  # "fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,  # if hparams.use_amp else 32,
        # stochastic_weight_avg=True,
        # deterministic=False,
        # profiler="simple",
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    train_label3d_folders = [
    ]

    train_image2d_folders = [
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = [
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    val_image2d_folders = [
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir,
                     'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(
            hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = NeRVDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        shape=hparams.shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = NeRVLightningModule(
        hparams=hparams
    )
    model = model.load_from_checkpoint(
        hparams.ckpt, strict=False) if hparams.ckpt is not None else model

    trainer.fit(
        model,
        datamodule,
    )

    # test

    # serve
