import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import pytorch3d

from ray_utils import (
    get_pixels_from_image,
    get_rays_from_pixels
)
import numpy as np

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

def get_mesh_renderer_with_depth(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius, bin_size=0)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def render_images(
    model,
    cameras,
    image_size,
    save=False,
    file_prefix='',
    lights=None,
    feat='color'
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        with torch.no_grad():
            torch.cuda.empty_cache()

            # Get rays
            camera = camera.to(device)
            light_dir = None
            # We assume the object is placed at the origin
            origin = torch.tensor([0.0, 0.0, 0.0], device=device) 
            light_location = None if lights is None else lights[cam_idx].location.to(device)
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
        all_images.append(image)

        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )
    
    return all_images

def render_pointcloud(point_cloud, image_size=256, background_color=[1,1,1], device=None):
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    images = []
    n_frames = 30
    a = np.pi/2
    #R_relative = [[1, 0, 0],
    #[0, np.cos(a), -np.sin(a)], 
    #[0, np.sin(a), np.cos(a)]]
    R_relative = [[np.cos(a), -np.sin(a), 0],
    [np.sin(a), np.cos(a), 0], 
    [0, 0, 1 ]]
    for i in range(n_frames):
        elev = 30 * np.pi / 180
        #azim = np.sin(4 * np.pi*i/n_frames)
        azim = 2 * np.pi*i/n_frames
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=6, elev=elev, azim=azim, degrees=False, up=[[0,1,0]])

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        # R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
        # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        images.append(rend)
    return images

def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    # assert image.shape[0] == image.shape[1], "Image must be square."
    # image_shape = image.shape[0]
    ndc_pixel_coordinatesY = torch.linspace(1, -1, image.shape[0]).to(device)
    ndc_pixel_coordinatesX = torch.linspace(1, -1, image.shape[1]).to(device)
    Y, X = torch.meshgrid(ndc_pixel_coordinatesY, ndc_pixel_coordinatesX)
    # print("X shape: ", X.shape)
    # print("Y shape: ", Y.shape)
    # print("depth shape: ", depth.shape)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb

def unproject_depth_image_old(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape).to(device)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    # print("X shape: ", X.shape)
    # print("Y shape: ", Y.shape)
    # print("depth shape: ", depth.shape)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb