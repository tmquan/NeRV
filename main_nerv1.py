import os
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torch.set_float32_matmul_precision('medium')

from typing import Optional
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer
from nerv.renderer import NeRVFrontToBackInverseRenderer, NeRVFrontToBackFrustumFeaturer, make_cameras

class NeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.gan = hparams.gan
        self.img = hparams.img
        self.vol = hparams.vol
        self.cam = hparams.cam
        self.sup = hparams.sup
        self.ckpt = hparams.ckpt
        self.strict = hparams.strict
        
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.lambda_gp = hparams.lambda_gp
        self.clamp_val = hparams.clamp_val
       
        self.logsdir = hparams.logsdir
       
        self.sh = hparams.sh
        self.pe = hparams.pe
        
        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.backbone = hparams.backbone
        self.devices = hparams.devices
        
        self.save_hyperparameters()

        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.img_shape, 
            image_height=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=8.0, 
            max_depth=12.0, 
            ndc_extent=3.0,
        )
        
        self.inv_renderer = NeRVFrontToBackInverseRenderer(
            in_channels=1, 
            out_channels=self.sh**2 if self.sh>0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            sh=self.sh, 
            pe=self.pe,
            backbone=self.backbone,
        )
        
        if self.ckpt:
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)
            
        if self.cam:
            self.cam_settings = NeRVFrontToBackFrustumFeaturer(
                in_channels=1, 
                out_channels=2, 
                backbone=self.backbone,
            )
            torch.nn.init.trunc_normal_(self.cam_settings.model._fc.weight.data, mean=0.0, std=0.05, a=-0.05, b=0.05)
            torch.nn.init.trunc_normal_(self.cam_settings.model._fc.bias.data, mean=0.0, std=0.05, a=-0.05, b=0.05)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.loss = nn.L1Loss(reduction="mean")
    
    def forward_screen(self, image3d, cameras, is_training=True):   
        with torch.set_grad_enabled(True and is_training):
            return self.fwd_renderer(image3d, cameras) 

    def forward_volume(self, image2d, elev, azim, n_views=[2, 1], is_training=True): 
        with torch.set_grad_enabled(self.vol and is_training):     
            return self.inv_renderer(image2d * 2.0 - 1.0, elev.squeeze(1), azim.squeeze(1), n_views) #* 0.5 + 0.5

    def forward_camera(self, image2d, is_training=True):
        with torch.set_grad_enabled(self.cam and is_training):    
            return self.cam_settings(image2d * 2.0 - 1.0)

    def forward_critic(self, image2d):
        return self.critic_model(image2d * 2.0 - 1.0)
    
    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]
            
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        src_elev_random = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.batch_size]).to(_device) #[-1 1]
        src_azim_random = torch.rand_like(src_elev_random) * 2 - 1 # [0 1) to [-1 1)
        src_dist_random = 10.0 * torch.ones(self.batch_size, device=_device)
        camera_random = make_cameras(src_dist_random, src_elev_random, src_azim_random)
        
        src_elev_locked = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.batch_size]).to(_device) #[-1 1]
        src_azim_locked = torch.rand_like(src_elev_locked) * 2 - 1 # [0 1) to [-1 1)
        src_dist_locked = 10.0 * torch.ones(self.batch_size, device=_device)
        camera_locked = make_cameras(src_dist_locked, src_elev_locked, src_azim_locked)

        est_figure_ct_random = self.forward_screen(image3d=image3d, cameras=camera_random)
        est_figure_ct_locked = self.forward_screen(image3d=image3d, cameras=camera_locked)
        
        est_dist_random = 10.0 * torch.ones(self.batch_size, device=_device)
        est_dist_locked = 10.0 * torch.ones(self.batch_size, device=_device)
        
        # XR pathway
        src_figure_xr_hidden = image2d
        est_dist_hidden = 10.0 * torch.ones(self.batch_size, device=_device)

        if self.cam:   
            if self.img:     
                # Reconstruct the cameras
                est_feat_random, \
                est_feat_locked, \
                est_feat_hidden = torch.split(
                    self.forward_camera(
                        image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden])
                    ), self.batch_size
                )

                est_elev_random, est_azim_random = torch.split(est_feat_random, 1, dim=1)
                est_elev_locked, est_azim_locked = torch.split(est_feat_locked, 1, dim=1)
                est_elev_hidden, est_azim_hidden = torch.split(est_feat_hidden, 1, dim=1)
            else:
                # Reconstruct the cameras
                est_feat_random, \
                est_feat_locked = torch.split(
                    self.forward_camera(
                        image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked])
                    ), self.batch_size
                )

                est_elev_random, est_azim_random = torch.split(est_feat_random, 1, dim=1)
                est_elev_locked, est_azim_locked = torch.split(est_feat_locked, 1, dim=1)
                
        else:
            if self.img:  
                est_elev_random, est_azim_random = src_azim_random, src_elev_random 
                est_elev_locked, est_azim_locked = src_azim_locked, src_elev_locked 
                est_elev_hidden, est_azim_hidden = torch.zeros(self.batch_size, device=_device), torch.zeros(self.batch_size, device=_device)
            else:
                est_elev_random, est_azim_random = src_azim_random, src_elev_random 
                est_elev_locked, est_azim_locked = src_azim_locked, src_elev_locked 
                
        if self.sup:
            camera_random = make_cameras(src_dist_random, src_elev_random, src_azim_random)
            camera_locked = make_cameras(src_dist_locked, src_elev_locked, src_azim_locked)
            if self.img:
                camera_hidden = make_cameras(est_dist_hidden, est_elev_hidden, est_azim_hidden)
        else:
            camera_random = make_cameras(est_dist_random, est_elev_random, est_azim_random)
            camera_locked = make_cameras(est_dist_locked, est_elev_locked, est_azim_locked)
            if self.img:
                camera_hidden = make_cameras(est_dist_hidden, est_elev_hidden, est_azim_hidden)

    
        cam_view = [self.batch_size, 1]     
        if self.img:
            est_volume_ct_random, \
            est_volume_ct_locked, \
            est_volume_xr_hidden = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden]),
                    elev=torch.cat([est_elev_random.view(cam_view), est_elev_locked.view(cam_view), est_elev_hidden.view(cam_view)]),
                    azim=torch.cat([est_azim_random.view(cam_view), est_azim_locked.view(cam_view), est_azim_hidden.view(cam_view)]),
                    n_views=[2, 1],
                    is_training=(stage=='train')
                ), self.batch_size
            )  
        else:
            est_volume_ct_random, \
            est_volume_ct_locked = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked]),
                    elev=torch.cat([est_elev_random.view(cam_view), est_elev_locked.view(cam_view)]),
                    azim=torch.cat([est_azim_random.view(cam_view), est_azim_locked.view(cam_view)]),
                    n_views=[2],
                    is_training=(stage=='train')
                ), self.batch_size
            ) 
        
            
        # Reconstruct the appropriate XR
        rec_figure_ct_random = self.forward_screen(image3d=est_volume_ct_random, cameras=camera_random, is_training=(stage=='train'))
        rec_figure_ct_locked = self.forward_screen(image3d=est_volume_ct_locked, cameras=camera_locked, is_training=(stage=='train'))
        if self.img:
            est_figure_xr_hidden = self.forward_screen(image3d=est_volume_xr_hidden, cameras=camera_hidden, is_training=(stage=='train'))

        # Perform Post activation like DVGO      
        est_volume_ct_random = est_volume_ct_random.sum(dim=1, keepdim=True)
        est_volume_ct_locked = est_volume_ct_locked.sum(dim=1, keepdim=True)
        if self.img:
            est_volume_xr_hidden = est_volume_xr_hidden.sum(dim=1, keepdim=True)

        # Compute the loss
        im2d_loss_ct_random = self.loss(est_figure_ct_random, rec_figure_ct_random) 
        im2d_loss_ct_locked = self.loss(est_figure_ct_locked, rec_figure_ct_locked) 
        
        im3d_loss_ct_random = self.loss(image3d, est_volume_ct_random) #+ self.loss(image3d, mid_volume_ct_random) 
        im3d_loss_ct_locked = self.loss(image3d, est_volume_ct_locked) #+ self.loss(image3d, mid_volume_ct_locked) 

        im2d_loss_ct = im2d_loss_ct_random + im2d_loss_ct_locked    
        if self.img: 
            im2d_loss_xr_hidden = self.loss(src_figure_xr_hidden, est_figure_xr_hidden) 
            im2d_loss_xr = im2d_loss_xr_hidden 
            im2d_loss = im2d_loss_ct + im2d_loss_xr
        else:
            im2d_loss = im2d_loss_ct
        
        im3d_loss_ct = im3d_loss_ct_random + im3d_loss_ct_locked
        im3d_loss = im3d_loss_ct
        
        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)

        p_loss = self.gamma*im2d_loss + self.alpha*im3d_loss 
        
        if self.cam:
            view_loss_ct_random = self.loss(torch.cat([src_elev_random, src_azim_random]), 
                                            torch.cat([est_elev_random, est_azim_random]))
            view_loss_ct_locked = self.loss(torch.cat([src_elev_locked, src_azim_locked]), 
                                            torch.cat([est_elev_locked, est_azim_locked]))
            view_loss_ct = view_loss_ct_random + view_loss_ct_locked

            view_loss = view_loss_ct
            
            self.log(f'{stage}_view_loss', view_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            c_loss = self.gamma*im2d_loss + self.theta*view_loss 

        if batch_idx==0:
            viz2d = torch.cat([
                torch.cat([est_figure_ct_random, est_figure_ct_locked], dim=-2).transpose(2, 3),
                torch.cat([rec_figure_ct_random, rec_figure_ct_locked], dim=-2).transpose(2, 3),  
            ], dim=-2)
            if self.img:
                viz2d = torch.cat([
                    viz2d, 
                    torch.cat([image2d, est_figure_xr_hidden], dim=-2).transpose(2, 3)
                ], dim=-2)
                
            viz3d = torch.cat([
                    torch.cat([image3d[..., self.vol_shape//2, :], 
                                est_volume_ct_locked[..., self.vol_shape//2, :]], dim=-2).transpose(2, 3),
                ], dim=-2)
            if self.img:
                viz3d = torch.cat([
                        viz3d, 
                        est_volume_xr_hidden[..., self.vol_shape//2, :],
                    ], dim=-2)

            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            grid3d = torchvision.utils.make_grid(viz3d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            if self.img_shape==self.vol_shape:
                grid = torch.cat([grid2d, grid3d], dim=-2)
                tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)    
            else:
                tensorboard.add_image(f'{stage}_2d_samples', grid2d.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
                tensorboard.add_image(f'{stage}_3d_samples', grid3d.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        

        if not self.gan:
            if self.vol:
                return p_loss
            if self.cam:
                return p_loss + c_loss
            
        elif self.gan:
            optimizer_g, optimizer_d = self.optimizers()
            # generator loss
            self.toggle_optimizer(optimizer_g)
            fake_images = torch.cat([rec_figure_ct_random, rec_figure_ct_locked, est_figure_xr_hidden])
            fake_scores = self.forward_critic(fake_images)
            g_loss = torch.mean(-fake_scores)
            loss = p_loss + g_loss + c_loss
            self.log(f'{stage}_g_loss', g_loss, on_step=(stage=='train'), prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
            self.manual_backward(loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
            
            # discriminator
            for p in self.critic_model.parameters():
                p.data.clamp_(-self.clamp_val, self.clamp_val)
            self.toggle_optimizer(optimizer_d)
            real_images = torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden])
            real_scores = self.forward_critic(real_images)
            fake_images = torch.cat([rec_figure_ct_random, rec_figure_ct_locked, est_figure_xr_hidden])
            fake_scores = self.forward_critic(fake_images.detach())
            d_loss = torch.mean(-real_scores) + torch.mean(+fake_scores)
            loss = d_loss 
            self.log(f'{stage}_d_loss', d_loss, on_step=(stage=='train'), prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
            self.manual_backward(loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            return p_loss + c_loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._common_step(batch, batch_idx, optimizer_idx, stage='train')
        self.train_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, optimizer_idx=-1, stage='validation')
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f'train_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.train_step_outputs.clear()  # free memory
        
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f'validation_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        
    def configure_optimizers(self):
        if self.gan:
            # If --gan is set, optimize Unprojector, Camera as generator, and Discriminator with 2 optimizers
            optimizer_g = torch.optim.AdamW(list(self.inv_renderer.parameters()) 
                                          + list(self.cam_settings.parameters()), lr=self.lr, betas=(0.9, 0.999))
            optimizer_d = torch.optim.AdamW(self.critic_model.parameters(), lr=self.lr * 4, betas=(0.9, 0.999))
            scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 200], gamma=0.1)
            scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[100, 200], gamma=0.1)
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d] 
        if self.vol:
            optimizer = torch.optim.AdamW(self.inv_renderer.parameters(), lr=self.lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            return [optimizer], [scheduler]
        if self.cam:
            optimizer = torch.optim.AdamW(self.cam_settings.parameters(), lr=self.lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            return [optimizer], [scheduler]
        if self.img:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            return [optimizer], [scheduler]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    
    # Model arguments
    parser.add_argument("--n_pts_per_ray", type=int, default=400, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--st", type=int, default=1, help="with spatial transformer network")
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    
    parser.add_argument("--gan", action="store_true", help="whether to train with GAN")
    parser.add_argument("--img", action="store_true", help="whether to train with XR")
    parser.add_argument("--vol", action="store_true", help="whether to train with CT")
    parser.add_argument("--cam", action="store_true", help="train cam locked or hidden")
    parser.add_argument("--sup", action="store_true", help="train cam ct or not")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")
    
    parser.add_argument("--alpha", type=float, default=1., help="vol loss")
    parser.add_argument("--gamma", type=float, default=1., help="img loss")
    parser.add_argument("--delta", type=float, default=1., help="vgg loss")
    parser.add_argument("--theta", type=float, default=1., help="cam loss")
    parser.add_argument("--omega", type=float, default=1., help="cam cond")
    parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty")
    parser.add_argument("--clamp_val", type=float, default=.1, help="gradient discrim clamp value")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--backbone", type=str, default='efficientnet-b7', help="Backbone for network")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_gan{int(hparams.gan)}_vol{int(hparams.vol)}_cam{int(hparams.cam)}_sup{int(hparams.sup)}_img{int(hparams.img)}",
        # filename='epoch={epoch}-validation_loss={validation_loss_epoch:.2f}',
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True, 
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_gan{int(hparams.gan)}_vol{int(hparams.vol)}_cam{int(hparams.cam)}_sup{int(hparams.sup)}_img{int(hparams.img)}", 
        log_graph=True
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
            swa_callback if not hparams.gan else None,
        ],
        accumulate_grad_batches=4 if not hparams.gan else 1,
        strategy="auto", 
        precision=16 if hparams.amp else 32,
        # gradient_clip_val=0.01, 
        # gradient_clip_algorithm="value"
        # stochastic_weight_avg=True if not hparams.gan else False,
        # deterministic=False,
        profiler="advanced"
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
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
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
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
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
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
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = NeRVLightningModule(
        hparams=hparams
    )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    trainer.fit(
        model,
        # compiled_model,
        train_dataloaders=datamodule.train_dataloader(), 
        val_dataloaders=datamodule.val_dataloader(),
        # datamodule=datamodule,
        ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve