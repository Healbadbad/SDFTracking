# Various loss functions and related utilities
# Most content from assignment 3 
# TODO: Github link

import torch
import torch.nn.functional as F

def eikonal_loss(gradients):
    """Implements eikonal loss on Nx3 gradients.
    Tries to enforce the gradient of F to be 1

    Args:
        gradients (torch.Tensor): Nx3 gradients
    
    Returns:
        torch.Tensor: eikonal loss
    """

    grads_magnitude = torch.norm(gradients, dim=-1)
    # mse loss with 1's
    loss = torch.nn.functional.mse_loss(grads_magnitude, torch.ones_like(grads_magnitude))
    return loss

def sphere_loss(signed_distance, points, radius=1.0):
    return torch.square(signed_distance[..., 0] - (torch.norm(points, dim=-1) - radius)).mean()

def get_random_points(num_points, bounds, device):
    min_bound = torch.tensor(bounds[0], device=device).unsqueeze(0)
    max_bound = torch.tensor(bounds[1], device=device).unsqueeze(0)

    return torch.rand((num_points, 3), device=device) * (max_bound - min_bound) + min_bound

def select_random_points(points, n_points):
    points_sub = points[torch.randperm(points.shape[0])]
    return points_sub.reshape(-1, 3)[:n_points]