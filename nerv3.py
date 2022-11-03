from model import Unet
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
import resource
import torchvision
import torch
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

        self.unet_model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )
        self.numsteps = 12
        self.stepsize = 30
        self.loss = nn.SmoothL1Loss(reduction="mean", beta=0.01)

    def forward(self, image3d):
        pass

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        randomt = torch.randint(-180, 180, (1,), device=_device).long()
        random_ = randomt.repeat(self.batch_size)

        # Construct the fixed cameras
        dist0 = 3.0 * torch.ones_like(random_)
        elev0 = torch.zeros_like(random_)
        azim0 = torch.zeros_like(random_)
        R0, T0 = look_at_view_transform(dist=dist0, elev=elev0, azim=azim0)
        cameras0 = FoVPerspectiveCameras(R=R0, T=T0).to(_device)
        singular_images = self.visualizer.forward(image3d=image3d, cameras=cameras0)

        # Construct the random camera
        dist_ = 3.0 * torch.ones_like(random_)
        elev_ = torch.zeros_like(random_)
        azim_ = random_
        R_, T_ = look_at_view_transform(dist=dist_, elev=elev_, azim=azim_)
        cameras_ = FoVPerspectiveCameras(R=R_, T=T_).to(_device)
        explicit_images = self.visualizer.forward(image3d=image3d, cameras=cameras_)

        implicit_images = singular_images
        expected_images = image2d
        concated_inputs = torch.cat([implicit_images, expected_images], dim=0)
        concated_output = self.unet_model.forward(concated_inputs, +random_.repeat(2))
        concated_second = self.unet_model.forward(concated_output, -random_.repeat(2))

        implicit_output, expected_output = concated_output[:self.batch_size], concated_output[self.batch_size:]
        implicit_second, expected_second = concated_second[:self.batch_size], concated_second[self.batch_size:]

        # # # for t in tqdm(range(0, numsteps), desc = 'Sampling loop time step', total = numsteps):
        # # for t in tqdm(torch.arange(0, int(randomt)), desc='Running forward pass', total=int(randomt)):
        # #     concated_images = self.unet_model.forward(
        # #         concated_images, t.repeat(2*self.batch_size).to(_device))

        # # Predict the explicit image from singular
        # # implicit_images, expected_images = concated_images[:self.batch_size], concated_images[self.batch_size:]
        # # for t in tqdm(range(0, numsteps), desc = 'Sampling loop time step', total = numsteps):
        # for t in tqdm(torch.arange(start=0, end=int(randomt), step=self.stepsize), desc='Running forward pass', total=int(randomt/self.stepsize)):
        #     # implicit_images = self.unet_model.forward(implicit_images, t.repeat(self.batch_size).to(_device))
        #     # expected_images = self.unet_model.forward(expected_images, t.repeat(self.batch_size).to(_device))
        #     concated_images = self.unet_model.forward(concated_images, t.detach().repeat(2*self.batch_size).to(_device))
        # implicit_output, expected_output = concated_images[:self.batch_size], concated_images[self.batch_size:]

        if batch_idx == 0:
            viz2d = torch.cat([singular_images,
                               explicit_images,
                               implicit_output,
                               implicit_second, 
                               image2d,
                               expected_output,
                               expected_second
                               ], dim=-2).transpose(2, 3)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)

        loss = self.loss(explicit_images, implicit_output) + self.loss(concated_inputs, concated_second)
        info = {f'loss': loss}
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
