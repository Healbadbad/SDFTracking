# Generates a SDF dataset from a given object file

import argparse
import os

import pytorch3d
import torch
import numpy as np

from utils import get_device, get_mesh_renderer_with_depth #, load_cow_mesh
import imageio
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes



def generate_360_obj_data(obj_path=None, image_size=256, n_images=30, device=None):
    """Generates renderings, camera poses, and ground truth depth values for a given obj file.
    
    Args:
        obj: path to obj file
    
    Returns:

    """

    if device is None:
        device = get_device()

    if obj_path is None:
        # Generate an ico sphere
        mesh = ico_sphere(level=3, device=device)

        # Cube mesh
        cube_vertices = torch.Tensor([[-1,-1,-1], [-1, -1, 1], [1,-1,1], [1,-1,-1], #bottom 4 
            [-1,1,-1], [-1, 1, 1], [1,1,1], [1,1,-1], # top 4 
            ]).to(device)
        cube_faces = torch.Tensor([[0,1,2], [2,3,0], # bottom
        [0,1,4], [4,5,1], # left
        [2,3,6], [6,7,3], # right
        [0,3,7], [7,4,0], # down
        [1,2,6], [6,5,1], # up
        [4,5,6], [6,7,4] # top
        ]).to(device)
        # mesh = Meshes(verts=[cube_vertices], faces=[cube_faces])    

        if mesh.textures is None:
            verts = mesh.verts_padded()
            # device = mesh.device
            color = (0.5,0.5,1)
            textures = torch.ones_like(verts).to(device)  # (1, N_v, 3)
            textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
            mesh.textures = pytorch3d.renderer.TexturesVertex(textures)
    else:
        mesh = pytorch3d.io.load_objs_as_meshes([obj_path])
        mesh = mesh.to(device)
    renderer = get_mesh_renderer_with_depth(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fov = 60

    camera_poses = []

    rgb_data = []
    depth_data = []
    camera_pose_data = []
    n_frames = n_images
    for i in range(n_frames):
        # elev = 30 * np.pi / 180
        #
        elev = 0

        #azim = np.sin(4 * np.pi*i/n_frames)
        azim = 2 * np.pi*i/n_frames
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, elev=elev, azim=azim, degrees=False, at=[[0,0,0]])

        # Prepare the camera:
        # cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        #     R=R, T=T, fov=60, device=device
        # )
        cameras = pytorch3d.renderer.PerspectiveCameras(
            R=R, T=T, device=device
        )

        rgb, depth = renderer(mesh, cameras=cameras, lights=lights)
        # Just take first image in batch since we are only rendering one view at a time
        rgb = rgb[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        rgb_data.append(rgb)
        depth = depth[0, ..., :1].cpu().numpy()  # (N, H, W, 1)
        depth_data.append(depth)
        camera_pose_data.append({"R": R, "T": T})
    
    return rgb_data, depth_data, camera_pose_data

    # rgb, depth = 

def save_synthetic_data(data, out_name="synthetic_data"):
    """Save the synthetic as a npz file."""
    data_dir = "data"
    with open(os.path.join(data_dir, out_name + ".npz"), 'wb') as f:
        rgb_data, depth_data, camera_pose_data = data
        np.savez(f, rgb_data=rgb_data, depth_data=depth_data, camera_pose_data=camera_pose_data)
    
    
def vis_synthetic_data(data, out_name="synthetic_data"):
    """Shows the synthetic data as a gif.  """
    vis_dir = "images"
    rgb_data, depth_data, camera_pose_data = data
    imageio.mimsave(os.path.join(vis_dir, out_name + "_rgb.gif"), rgb_data, fps=15)
    imageio.mimsave(os.path.join(vis_dir, out_name + "_depth.gif"), depth_data, fps=15)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_images", type=int, default=30)
    parser.add_argument("--out_name", type=str, default="synthetic_data")
    args = parser.parse_args()

    print("Generating synthetic data...")
    data = generate_360_obj_data(obj_path=args.obj_path, image_size=args.image_size)
    print("Visualizing synthetic data...")
    vis_synthetic_data(data, out_name=args.out_name)
    save_synthetic_data(data, out_name=args.out_name)

