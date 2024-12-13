
from sampler import StratifiedRaysampler
from renderer import SphereTracingRenderer, VolumeSDFRenderer
import torch

import argparse

class SDFRenderModel(torch.nn.Module):
    def __init__(
        self,
        sdf,
        # sampler,
        # renderer,
    ):
        super().__init__()
        # Get implicit function from config
        self.implicit_fn = sdf

        # Point sampling (raymarching) scheme
        self.sampler = StratifiedRaysampler(n_pts_per_ray=128, min_depth=0.0, max_depth=5.0)

        # Initialize implicit renderer
        render_cfg = argparse.Namespace()
        render_cfg.chunk_size = 32768 # not the same as batch size
        # render_cfg.chunk_size = int(32768/8) # not the same as batch size
        render_cfg.near = 0.0
        render_cfg.far = 5.0 # TODO: Make this a parameter
        render_cfg.max_iters = 64
        # self.renderer = SphereTracingRenderer(render_cfg)
        render_cfg.alpha = 10
        render_cfg.beta = 0.05

        self.renderer = VolumeSDFRenderer(render_cfg)
    
    def forward(
        self,
        ray_bundle,
        light_dir=None
    ):
        # Call renderer with
        #  a) Implicit function
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle,
            light_dir
        )