
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.



import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d.utils
import numpy as np

class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(
        self,
    ) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

def load_npz_rgbd_dataset(filename):
    data = np.load(filename, allow_pickle=True)
    rgb_data = data["rgb_data"].astype(np.float32)
    print("max", np.max(data["depth_data"]))
    print("min", np.min(data["depth_data"]))

    depth_data = data["depth_data"].astype(np.float32)
    print("max", np.max(depth_data))
    print("min", np.min(depth_data))
    camera_pose_data = data["camera_pose_data"]
    # print("camera_pose_data", camera_pose_data)
    n_cameras = len(camera_pose_data)
    cameras = [
        PerspectiveCameras(
            #**{k: v[cami][None] for k, v in camera_pose_data.items()}
            R=cami["R"], T=cami["T"]
        ).to("cpu")
        for cami in camera_pose_data
    ]

    idx = range(len(rgb_data))
    dataset = ListDataset(
        [
            {"image": torch.FloatTensor(rgb_data[i]),
             "depth_image": torch.FloatTensor(depth_data[i]),
             "camera": cameras[i],
             "camera_idx": int(i)}
            for i in idx
        ]
    )
    return dataset

def load_npz_rgbd_mask_dataset(filename):
    data = np.load(filename, allow_pickle=True)
    rgb_data = data["rgb_data"].astype(np.float32)
    print("max", np.max(data["depth_data"]))
    print("min", np.min(data["depth_data"]))
    mask_data = data["mask_data"].astype(np.float32)

    depth_data = data["depth_data"].astype(np.float32)
    print("max", np.max(depth_data))
    print("min", np.min(depth_data))
    camera_pose_data = data["camera_pose_data"]
    # print("camera_pose_data", camera_pose_data)
    n_cameras = len(camera_pose_data)
    # cameras = [
    #     PerspectiveCameras(
    #         #**{k: v[cami][None] for k, v in camera_pose_data.items()}
    #         R=cami["R"], T=cami["T"]
    #     ).to("cpu")
    #     for cami in camera_pose_data
    # ]
    # Convert opencv cameras to pytorch3d cameras
    f = 624.45429
    cx = 640
    cy = 360
    k = 0.019158445755302463
    camera_matrix = torch.tensor([[[f/k, 0, 0],
                                  [0, f/k, 0],
                                  [cx, cy, 1]]]).transpose(1, 2)

    cameras = [
        pytorch3d.utils.cameras_from_opencv_projection(
            R=cami["R"], tvec=cami["T"], camera_matrix=camera_matrix, image_size=torch.tensor([[720, 1280]])
        ).to("cpu")
        for cami in camera_pose_data
    ]

    idx = range(len(rgb_data))
    dataset = ListDataset(
        [
            {"image": torch.FloatTensor(rgb_data[i][..., 0:3] / 255.0),
             "depth_image": torch.FloatTensor(depth_data[i] / (2**10 - 1)),
             "camera": cameras[i],
             "camera_idx": int(i),
             "mask": torch.FloatTensor(mask_data[i]),
             }
            for i in idx
        ] [0:10]
    )
    return dataset