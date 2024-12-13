import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth,
        max_depth,
        
    ):
        super().__init__()

        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # print("n_points_per_ray", self.n_pts_per_ray)
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).cuda()
        # print("z_vals", z_vals.shape)
        # z_vals should be of shape (ray_bundle.origins.shape[0], self.n_pts_per_ray)
        z_vals = z_vals.unsqueeze(-1).expand(self.n_pts_per_ray, 3).cuda()
        # print("z_vals", z_vals.shape)

        # TODO (1.4): Sample points from z values
        # sample_points should be of shape (ray_bundle.origins.shape[0], 1, 3)
        # sample_lengths should be of shape (ray_bundle.origins.shape[0], 1, 3)
        # print("ray_bundle.origins", ray_bundle.origins.shape)
        # print("ray_bundle.directions", ray_bundle.directions.shape)

        #sample_points = ray_bundle.origins + ray_bundle.directions * z_vals
        directions = ray_bundle.directions.unsqueeze(1) * z_vals
        sample_points = ray_bundle.origins.unsqueeze(1) + directions
        # take from (n_rays, 3) to (n_rays, 1, 3)
        #sample_points.unsqueeze_(1)
        
        # print("sample_points", sample_points.shape)
        a = torch.ones_like(sample_points[..., :1])
        ray_bundle._replace(
                    sample_points=sample_points,
                    sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
                   # sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}