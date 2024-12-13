# Trains an SDF model on the given dataset and saves it to disk.
# Heavily based on SDF training code from assignment 3 of 16-825.
# (main.py)
# This uses RGB-D images recorded from a kinect, and camera poses generated using colmap

# The steps are:
# 1) Load the dataset
# 2.) Initialize the model and pretrain it with a sphere loss
# 3.) Train the model with the data using eikonal loss

import argparse
import os
import torch
import numpy as np
import tqdm
import imageio

from sdf.sdf import NeuralSurface
from sdf.sdf_renderer import SDFRenderModel

from sdf.losses import eikonal_loss, sphere_loss, get_random_points, select_random_points

from render_functions import render_geometry
from render_functions import render_points

from sampler import StratifiedRaysampler
from renderer import SphereTracingRenderer, VolumeSDFRenderer
import pytorch3d

from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_random_pixels_from_mask,
    get_rays_from_pixels
)

from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    create_surround_lights,
    vis_grid,
    vis_rays,
)

from utils import render_images, unproject_depth_image, render_pointcloud
from dataset import load_npz_rgbd_mask_dataset
import numpy as np

# Model class containing:
#   1) Implicit function defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit function given a sampling scheme


scale = 12
def pretrain_sdf(
    model,
    lr=5e-4,
    batch_size=1024,
    training_bounds=[[-scale, -scale, -scale], [scale, scale, scale]],
    pretrain_iters=1000,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # Run the main training loop.
    for iter in range(0, pretrain_iters):
        points = get_random_points(
            batch_size, training_bounds, 'cuda'
        )

        # Run model forward
        distances = model.implicit_fn.get_distance(points)
        eikonal_points = get_random_points(
            batch_size, training_bounds, 'cuda'
        )
        eikonal_distances, eikonal_gradients = model.implicit_fn.get_distance_and_gradient(eikonal_points)
        loss = sphere_loss(distances, points, 5.0)
        loss += eikonal_loss(eikonal_gradients) * .02 #training_cfg.eikonal_weight 

        # Take the training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_sdf(data_name, out_name):
    """Trains an SDF model on the given dataset and saves it to disk.
    This version of training uses RGB-D images with known camera poses to train the model.

    Args:
        data_name: name of the dataset to train on
        out_name: name of the output model
    """
    batch_size = 1024
    #training_bounds = [[-4, -4, -4], [4, 4, 4]]
    training_bounds = [[-scale, -scale, -scale], [scale, scale, scale]]
    # if not os.path.exists(out_name):
    #     os.makedirs(out_name) # Create the output directory if it doesn't exist
        
    # Create the dataloader
    train_dataset = load_npz_rgbd_mask_dataset(data_name)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )


    # Initialize the model
    sdf = NeuralSurface(None)
    sdf_render_model = SDFRenderModel(sdf)
    sdf_render_model.cuda()
    sdf_render_model.train()
    # Pretrain SDF
    print("Pretraining the SDF")
    pretrain_sdf(sdf_render_model, lr=5e-4, batch_size=batch_size)
    # test_images = render_images(
    #     sdf_render_model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
    #     (256, 256), file_prefix='volsdf'
    # )
    # imageio.mimsave(f'images_kinect/{out_name}_pretraining_progress.gif', [np.uint8(im * 255) for im in test_images])
    print("Pretrained the SDF")

    # Initialize the optimizer.
    training_lr = 5e-4
    optimizer = torch.optim.Adam(
        sdf.parameters(),
        lr=training_lr,
    )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    scheduler_gamma = 0.8
    scheduler_step_size = 50
    def lr_lambda(epoch):
        return scheduler_gamma ** (
            epoch / scheduler_step_size
        )
    # def lr_lambda(epoch):
    #     return 1

    start_epoch = 0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    training_cfg = argparse.Namespace()
    training_cfg.inter_weight = 0.1
    training_cfg.eikonal_weight = 0.02
    training_cfg.checkpoint_interval = 50
    training_cfg.checkpoint_path = "./sdf_checkpoints/"
    training_cfg.render_interval = 5
    checkpoint_path = os.path.join(training_cfg.checkpoint_path, out_name + ".pth")

    num_epochs = 2000
    image_size = (256, 256)
    # Run the main training loop.
    for epoch in range(0, num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            # print("batch:", batch)
            image, depth, camera, camera_idx, mask = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()
            depth = depth.cuda()
            print("max", torch.max(depth))
            print("min", torch.min(depth))
            print("camera location:", camera.T)

            print("mask min:", torch.min(mask))
            print("mask max:", torch.max(mask))

            # Depth images are mirrored in the y direction, so we need to correct it
            # depth = depth.flip(0)

            # Generate a mask from the depth image
            # mask = torch.zeros(depth.shape).cuda()
            mask = torch.where(mask > 0, 1, 0).cuda()
            mask = mask * depth # Ignore zero depth values
            depth = torch.where(depth < 0.001, -1, depth)
            mask = torch.where(mask > 0, 1, 0).cuda()
            imageio.mimsave(f"images_kinect/{out_name}_rgb_gt.gif", [np.uint8(image[0].cpu().numpy() * 255)])
            imageio.mimsave(f"images_kinect/{out_name}_mask_gt.gif", [np.uint8(mask.cpu().numpy() * 255)])
            imageio.mimsave(f"images_kinect/{out_name}_gt_depth.gif", [np.uint8(depth.cpu().numpy() * 255)])

            # imageio.mimsave(f"images_kinect/{out_name}_gt_depth.gif", [depth.cpu().numpy()])

            # Convert depth to distance / pointcloud
            points, rgb = unproject_depth_image(image[0], mask.squeeze(-1), depth, camera)
            distances, gradients = sdf_render_model.implicit_fn.get_distance_and_gradient(points)
            loss = 0

            point_loss = torch.nn.functional.mse_loss(distances, torch.zeros_like(distances))
            loss += point_loss
            print("point_loss: ", point_loss)





            # Sample rays
            xy_grid = get_random_pixels_from_image(
                batch_size, image_size, camera
            )
            # xy_grid = get_random_pixels_from_mask(
            #     batch_size, image_size, mask, camera
            # )
            print("xy_grid:", xy_grid.shape, xy_grid)
            print("camera:", camera)
            ray_bundle = get_rays_from_pixels(
                xy_grid, image_size, camera
            )
            rgb_gt = sample_images_at_xy(image, xy_grid)
            print("ray bundle shape:", ray_bundle.shape)
            print("rgb_gt shape:", rgb_gt.shape)

            # print("ray_bundle:", ray_bundle)
            # print("ray_bundle.directions:", ray_bundle.directions)
            # Run model forward
            out = sdf_render_model(ray_bundle)
            print("out shape:", out['color'].shape)

            # Color loss
            image_loss = torch.mean(torch.square(rgb_gt - out['color']))
            loss += image_loss
            print("image_loss:", image_loss)

            # Sample random points in bounding box
            eikonal_points = get_random_points(
                batch_size, training_bounds, 'cuda'
            )

            # Get sdf gradients and enforce eikonal loss
            eikonal_distances, eikonal_gradients = sdf_render_model.implicit_fn.get_distance_and_gradient(eikonal_points)
            loss += torch.exp(-1e2 * torch.abs(eikonal_distances)).mean() * training_cfg.inter_weight
            loss += eikonal_loss(eikonal_gradients) * training_cfg.eikonal_weight 

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {image_loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % training_cfg.checkpoint_interval == 0
            and len(training_cfg.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": sdf_render_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % training_cfg.render_interval == 0
            and epoch > 0
        ):
            # Save a debug pointcloud image
            verts = points.to(camera.device).unsqueeze(0)
            rgb = rgb.to(camera.device).unsqueeze(0)
            point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
            print("before pointcloud render")
            pointcloud_images = render_pointcloud(point_cloud, device=camera.device)
            imageio.mimsave(f'images_kinect/{out_name}_pointcloud.gif', [np.uint8(im * 255) for im in pointcloud_images])
            print("after pointcloud render")

            # Save a debug RGB image
            imageio.mimsave(f"images_kinect/{out_name}_rgb_gt.gif", [np.uint8(image[0].cpu().numpy() * 255)])
            print("before sdf render")
            test_images = render_images(
                sdf_render_model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0),
                image_size, file_prefix='volsdf'
            )
            imageio.mimsave(f'images_kinect/{out_name}_training_progress.gif', [np.uint8(im * 255) for im in test_images])
            print("after sdf render")

            try:
                test_images = render_geometry(
                    sdf_render_model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0),
                    image_size, file_prefix='volsdf_geometry'
                )
                imageio.mimsave(f'images_kinect/{out_name}_training_geometry.gif', [np.uint8(im * 255) for im in test_images])
            except Exception as e:
                print("Empty mesh")
                pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="data/cow.npz")
    parser.add_argument("--out_name", type=str, default="sdf_trained/synthetic_data.sdf")
    args = parser.parse_args()

    train_sdf(args.data_name, args.out_name)
