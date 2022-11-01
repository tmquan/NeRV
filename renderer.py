import torch
import torch.nn as nn
from pytorch3d.renderer import VolumeRenderer

class FigureRenderer(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer
        
    def forward(self, cameras, volumes, norm_type="standardized", eps=1e-8):
        # screen_RGBA, ray_bundles = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]
        # rays_points = ray_bundle_to_ray_points(ray_bundles)
        screen_RGBA, _ = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1) # 3 for NeRF
        screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        minimized = lambda x: (x + eps)/(x.max() + eps)
        normalized = lambda x: (x - x.min() + eps)/(x.max() - x.min() + eps)
        standardized = lambda x: (x - x.mean())/(x.std() + 1e-4) # 1e-6 to avoid zero division
        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        return screen_RGB
