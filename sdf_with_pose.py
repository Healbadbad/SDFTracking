# A class that keeps track of the pose for an sdf
# This class will be used to estimate the location of the sdf in an image

import torch
from torch import nn
from pytorch3d.transforms import euler_angles_to_matrix, Rotate

class PoseSDF(nn.Module):
    def __init__(self, sdf):
        super().__init__()
        # self.center = nn.Parameter(torch.tensor([0,0,0]).float().unsqueeze(0), requires_grad=True)
        # TODO: Refactor back to sdf
        self.implicit_fn = sdf
        self.pose = nn.Parameter(torch.tensor([0,0,0,0,0,0]).float().unsqueeze(0), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor([1,1,1]).float().unsqueeze(0), requires_grad=True)
        self.rot = nn.Parameter(torch.tensor([0,0,0]).float().unsqueeze(0), requires_grad=True)
    
    def pose_parameters(self,):
        return self.pose, self.scale

    def translation_parameters(self,):
        return [self.pose]

    def rotation_parameters(self,):
        return [self.rot]
    
    def transform_points(self, points):
        expanded_pose = self.pose.expand(points.shape[0], -1)
        expanded_scale = self.scale.expand(points.shape[0], -1)
        if len(points.shape) == 3:
            expanded_pose = self.pose.expand(points.shape[0], points.shape[1], -1)
            expanded_scale = self.scale.expand(points.shape[0], points.shape[1], -1)

        # expanded_pose = self.pose.expand(points.shape)
        # expanded_scale = self.scale.expand(points.shape)
        #transformed_points = points * expanded_scale + expanded_pose[:, :3]
        R = euler_angles_to_matrix(self.rot, "XYZ")
        # TODO: Rotate correctly
        rotato = Rotate(R)
        rotated_points = rotato.transform_points(points)
        transformed_points = rotated_points - expanded_pose[... , :3]
        #transformed_points = points * expanded_scale + expanded_pose
        return transformed_points
    
    def get_distance_color(self, points):
        # print("get_distance_color points:", points.shape)
        transformed_points = self.transform_points(points)
        return self.implicit_fn.get_distance_color(transformed_points)

    
    def get_distance(self, points):
        """ Computes the distance between the given points and the sdf given it's position
        
        Args:
            points (torch.Tensor): A tensor of shape (N, 3) or containing the points to compute the distance to

        """
        transformed_points = self.transform_points(points)
        return self.implicit_fn(transformed_points)
    
    def forward(self, points):
        """ Computes the distance between the given points and the sdf given it's position
        
        Args:
            points (torch.Tensor): A tensor of shape (N, 3) containing the points to compute the distance to

        """
        transformed_points = self.transform_points(points)
        return self.implicit_fn(transformed_points)