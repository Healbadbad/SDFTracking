import cv2
import numpy as np
import os

import pykinect_azure as pykinect
from utils import Open3dVisualizer

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
	# print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	# Initialize the Open3d visualizer
	open3dVisualizer = Open3dVisualizer()

	cv2.namedWindow('Transformed color',cv2.WINDOW_NORMAL)
	while True:

		# Get capture
		capture = device.update()

		# Get the 3D point cloud
		#ret_point, points = capture.get_transformed_pointcloud()
		#ret_point, colored_depth = capture.get_transformed_colored_depth_image()
		ret_point, transformed_color = capture.get_transformed_color_image()
		ret_point, transformed_depth = capture.get_transformed_depth_image()

		# Get the color image in the depth camera axis
		ret_color, color_image = capture.get_color_image()
		from vic import vic
		vic.interact(locals())

		if not ret_color or not ret_point:
			continue

		# open3dVisualizer(points, color_image)
		# print("colored_depth.shape", colored_depth.shape)
		# cv2.imshow('Transformed color', colored_depth[:,:,0:3])
		cv2.imshow('Transformed depth', transformed_depth)

		# cv2.imshow('Transformed color', transformed_color_image)
		cv2.imshow('Transformed color', transformed_color)
		
		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):  
			break



def save_synthetic_data(data, out_name="synthetic_data"):
    """Save the synthetic as a npz file."""
    data_dir = "data"
    with open(os.path.join(data_dir, out_name + ".npz"), 'wb') as f:
        rgb_data, depth_data, camera_pose_data = data
        np.savez(f, rgb_data=rgb_data, depth_data=depth_data, camera_pose_data=camera_pose_data)