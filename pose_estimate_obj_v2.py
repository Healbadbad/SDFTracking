# Loads an obj file and uses an SDF to estimate its pose

import argparse
import os

import pytorch3d
import torch
import numpy as np

from utils import get_device, get_mesh_renderer_with_depth #, load_cow_mesh
from utils import render_images, unproject_depth_image, render_pointcloud
from sdf.losses import eikonal_loss, sphere_loss, get_random_points, select_random_points

import imageio
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from sdf_with_pose import PoseSDF
from sdf.sdf import NeuralSurface
from sdf.sdf_renderer import SDFRenderModel
from copy import deepcopy

from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    create_surround_lights,
    vis_grid,
    vis_rays,
)

from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_random_pixels_from_mask,
    get_rays_from_pixels
)

from pytorch3d.renderer import (
    PerspectiveCameras,
)

def estimate_pose(obj_path, pose_sdf, image_size=256, n_images = 60, vis=False, device=None):
    """Estimates the pose of an obj file using an SDF.
    Assumes the camera is always at 0,0,0 since we want a relative pose.

    Args:
        obj_path: path to obj file
        n_images: number of images to render and use for pose estimation

    """
    if device is None:
        device = get_device()
    pose_sdf = pose_sdf.to(device)
    pose_sdf_render_model = SDFRenderModel(pose_sdf)

    if obj_path is None:
        # Just create a sphere

        original_mesh = ico_sphere(level=3, device=device)
        if original_mesh.textures is None:
            verts = original_mesh.verts_padded()
            # device = mesh.device
            color = (0.5,0.5,1)
            textures = torch.ones_like(verts).to(device)  # (1, N_v, 3)
            textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
            original_mesh.textures = pytorch3d.renderer.TexturesVertex(textures)
    else:
        original_mesh = pytorch3d.io.load_objs_as_meshes([obj_path])
    original_mesh = original_mesh.to(device)
    renderer = get_mesh_renderer_with_depth(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)


    rgb_data = []
    depth_data = []
    camera_pose_data = []
    n_frames = n_images

    # TODO: Move into loop so every frame is independent
    print("parameters:", pose_sdf.parameters())
    lr = 1e-1
    # lr = 1
    # optimizer = torch.optim.SGD(
    #     pose_sdf.pose_parameters(),
    #     lr=lr,
    # )

    all_images_sdf = []
    all_images_gt = []


    start_pos = np.array([-2, -2, 2])
    end_pos = np.array([2, 2, 6])

    translation_history = []
    pose_estimation_history = []
    loss_history = []


    for i in range(n_frames):

        # offset = torch.tensor([1,1,5]).to(device)
        # Linearly interpolate between start and end position
        offset = torch.tensor(start_pos * (1 - i / n_frames) + end_pos * (i / n_frames)).to(device).float()
        translated_mesh = original_mesh.offset_verts(offset)
        # Prepare the camera:
        # cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        #     R=R, T=T, fov=60, device=device
        # )

        # initialize the camera at 0,0,0
        cameras = pytorch3d.renderer.PerspectiveCameras(
            device=device
        )
        camera = cameras[0]

        image, depth = renderer(translated_mesh, cameras=cameras, lights=lights)
        mask = torch.zeros(depth.shape).cuda()
        mask = torch.where(depth > 0, 1, 0)
        # print("image shape: {}".format(image.shape))
        # print("depth shape: {}".format(depth.shape))
        # print("mask shape: {}".format(mask.shape))

        points, rgb = unproject_depth_image(image[0][:,:,:3], mask[0].squeeze(-1), depth[0].squeeze(-1), cameras)

        # Initialize the pose at the centroid of the points
        # TODO: Only initialize for the first frame?
        if i == 0:
            centroid = torch.mean(points, dim=0)
            print("centroid shape: {}".format(centroid.shape))
            pose_sdf.pose.requires_grad = False
            # pose_sdf.pose[0, 0:3] += centroid
            pose_sdf.pose[0, 0:3] += centroid
            pose_sdf.pose.requires_grad = True

        optimizer_steps = 50

        optimizer = torch.optim.Adam(
            pose_sdf.pose_parameters(),
            lr=lr,
        )

        for j in range(optimizer_steps):
            # TODO: Move this out of the loop
            # and initialize pose to average position
            print("Pose:", pose_sdf.pose)
            #distances = pose_sdf(points[0:1])
            distances = pose_sdf(points)
            print("distances shape: {}".format(distances.shape))
            print("distances:", distances)
            loss = 0
            point_loss = torch.nn.functional.mse_loss(distances, torch.zeros_like(distances))
            print(f"point_loss: {point_loss}")

            loss += point_loss

            optimizer.zero_grad()
            loss.backward()
            print("pose:", pose_sdf.pose)
            print("pose grad:", pose_sdf.pose.grad)
            print("point", points[0])
            optimizer.step()
        loss_history.append(loss.detach().cpu().numpy())




        # Visualization
        if vis:
            focal_length = 1.0
            sdf_cam = PerspectiveCameras(
                focal_length=torch.tensor([focal_length])[None],
                principal_point=torch.tensor([0.0, 0.0])[None],
                # R=R,
                # T=T,
            )
            # sdf_cam = create_surround_cameras(0.001, n_poses=1, up=(0.0, 1.0, 0.0), focal_length=1.0)[0]
            #sdf_image = render_image(pose_sdf_render_model, camera, (image_size, image_size), feat='color')
            sdf_image = render_image(pose_sdf_render_model, sdf_cam, (image_size, image_size), feat='color')
            all_images_sdf.append(sdf_image)
            all_images_gt.append(image[0][:,:,:3].cpu().numpy())
        
        # Record trajectory and error information 
        translation_history.append(deepcopy(offset))
        pose_estimation_history.append(deepcopy(pose_sdf.pose))





        # Just take first image in batch since we are only rendering one view at a time
        # rgb = rgb[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        # rgb_data.append(rgb)
        # depth = depth[0, ..., :1].cpu().numpy()  # (N, H, W, 1)
        # depth_data.append(depth)
        # camera_pose_data.append({"R": R, "T": T})
    if vis:
        imageio.mimsave(f'images/pose_estimation_sdf.gif', [np.uint8(im * 255) for im in all_images_sdf])
        imageio.mimsave(f'images/pose_estimation_gt.gif', [np.uint8(im * 255) for im in all_images_gt])
    
    pose_error_history = []
    for gt_pose, estimated_pose in zip(translation_history, pose_estimation_history):
        pose_estimation_error = np.sum(estimated_pose[0:3] - gt_pose[0:3])
        
        pose_error_history.append(pose_estimation_error)

    # Plot losses and trajectory
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Timestep")
    plt.ylabel("SDF fit Loss")
    plt.savefig('images/loss_history.png')

    fig = plt.figure()
    plt.plot(pose_error_history)
    plt.xlabel("Timestep")
    plt.ylabel("L1 pose estimation error ")
    plt.savefig('images/L1_error_history.png')
    

def render_image(
    model,
    camera,
    image_size,
    save=False,
    file_prefix='',
    lights=None,
    feat='color'
):
    # all_images = []
    device = list(model.parameters())[0].device

    # for cam_idx, camera in enumerate(cameras):
    print(f'Rendering image')

    with torch.no_grad():
        torch.cuda.empty_cache()

        # Get rays
        camera = camera.to(device)
        light_dir = None
        # We assume the object is placed at the origin
        origin = torch.tensor([0.0, 0.0, 0.0], device=device) 
        light_location = None if lights is None else lights[0].location.to(device)
        if lights is not None:
            #light_dir = (origin - light_location) / torch.norm(origin - light_location) #TODO: Use light location and origin to compute light direction
            light_dir = origin - light_location #TODO: Use light location and origin to compute light direction
            light_dir = torch.nn.functional.normalize(light_dir, dim=-1).view(-1, 3)
        xy_grid = get_pixels_from_image(image_size, camera)
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)

        # Run model forward
        out = model(ray_bundle, light_dir)

    # Return rendered features (colors)
    image = np.array(
        out[feat].view(
            image_size[1], image_size[0], 3
        ).detach().cpu()
    )
    return image


def load_sdf(sdf_path):
    """Loads an SDF from a file.
    """
    # TODO: Load sdf from checkpoint
    loaded_data = torch.load(sdf_path)
    sdf = NeuralSurface(None)
    # print("loaded data: {}".format(loaded_data.keys()))
    # print("loaded data: {}".format(loaded_data["model"]))
    # print("loaded data: {}".format(loaded_data["model"].keys()))
    #sdf.load_state_dict(loaded_data["model"])
    pose_sdf = PoseSDF(sdf)
    pose_sdf.load_state_dict(loaded_data["model"], strict=False) # Strict is false since we have extra paramers in the PoseSDF
    # print("Loaded sdf")
    # start_epoch = loaded_data["epoch"]


    # print(f"   => resuming from epoch {start_epoch}.")
    # optimizer_state_dict = loaded_data["optimizer"]
    # TODO: Create PoseSDF
    return pose_sdf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sdf_path", type=str, default=None)
    parser.add_argument("--vis", action="store_true" )
    args = parser.parse_args()

    pose_sdf = load_sdf(args.sdf_path)
    estimate_pose(args.obj_path, pose_sdf, vis=args.vis)