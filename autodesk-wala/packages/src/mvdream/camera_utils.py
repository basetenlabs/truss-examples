import os

import numpy as np
import torch

from cloudpathlib import S3Path
from src.mvdream.constants import (
    BUILD3D_CAMERA_ROTATION,
    BUILD3D_CAMERA_ELEVATION,
)


def create_camera_to_world_matrix(elevation, azimuth):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.cos(azimuth)

    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender


def normalize_camera(camera_matrix):
    """normalize the camera location onto a unit-sphere"""
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            torch.norm(translation, dim=1, keepdim=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    return camera_matrix.reshape(-1, 16)


def get_camera(
    num_frames, elevation=15, azimuth_start=0, azimuth_span=360, blender_coord=True
):
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span + azimuth_start, angle_gap):
        print(elevation, azimuth)
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
    return torch.tensor(np.stack(cameras, 0)).float()


def get_camera_build3d(indices_list, blender_coord=True):
    rotation_camera = np.array(BUILD3D_CAMERA_ROTATION)
    elevation_camera = np.array(BUILD3D_CAMERA_ELEVATION)
    assert len(rotation_camera) == len(elevation_camera)

    cameras = []
    for i in indices_list:  # range(len(rotation_camera)):#indices_list:
        elevation = elevation_camera[i]
        azimuth = rotation_camera[i]
        print(i, elevation, azimuth)
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
    return torch.tensor(np.stack(cameras, 0))  # .float()


def get_camera_objaverse(camera_path, blender_coord=True):
    cp = S3Path("s3://mvdream-training-bucket-ue1/data/rotation.npy")
    cp.download_to("./rotation.npy")
    cp = S3Path("s3://mvdream-training-bucket-ue1/data/elevation.npy")
    cp.download_to("./elevation.npy")
    rotation_camera = np.load("./rotation.npy")
    elevation_camera = np.load("./elevation.npy")
    assert len(rotation_camera) == len(elevation_camera)

    cameras = []
    for i in range(len(rotation_camera)):
        elevation = elevation_camera[i]
        azimuth = rotation_camera[i]
        # print(i, elevation, azimuth)
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
    return torch.tensor(np.stack(cameras, 0))  # .float()
