import functools
import uuid
import tempfile

import h5py
import numpy as np
import os.path as osp
import torch
import json

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.v2 import (
    ColorJitter,
    RandomAffine,
)


def load_arr_file(file, keys, file_type):
    assert file_type in ["h5df", "npz"]

    ### results
    results = {}
    if file_type == "h5df":
        arr = h5py.File(file, "r")

        for key in keys:
            data = np.array(arr[key][:])
            results[key] = data
    elif file_type == "npz":
        arr = np.load(file)

        for key in keys:
            data = np.array(arr[key])
            results[key] = data
    else:
        raise Exception(f"Unknown file type for {file_type}....")

    return results


def make_a_grid(occupancy_arr, color_numpy, voxel_resolution=32):
    voxel_resolution = int(voxel_resolution)
    cube_color = np.zeros(
        [voxel_resolution + 1, voxel_resolution + 1, voxel_resolution + 1, 4]
    )
    cube_color[occupancy_arr[:, 0], occupancy_arr[:, 1], occupancy_arr[:, 2], 0:3] = (
        color_numpy
    )
    cube_color[occupancy_arr[:, 0], occupancy_arr[:, 1], occupancy_arr[:, 2], 3] = 1
    return cube_color[:-1, :-1, :-1]


def load_image(path, image_over_white=None):
    """
    Returns a PIL.Image in RGB mode.

    If the input image has an alpha channel it will be replaced by black (default); see args for
    alternatives.

    Arguments:
        path:               A local path to an image file.
        image_over_white:   (optional) Pass True to overlay the input image over a white background
                            when removing the alpha channel. Pass False to overlay over a black
                            background. Pass None to autodetect based on the given image's mode -
                            greyscale images are overlaid on white and RGB images are overlaid on
                            black.
    """
    img = Image.open(path)

    if image_over_white is None:
        if img.mode == "LA":
            image_over_white = True

    if image_over_white:
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, "WHITE")
        img = Image.alpha_composite(white, img).convert("RGB")
    else:
        img = img.convert("RGB")  # default behaviour is to overlay over black

    return img


def get_s3_object(s3_path):
    bucket = s3_path.split("/")[2]
    file_key = "/".join(s3_path.split("/")[3:])
    s3_client = get_s3_client("s3")
    filebuffer = s3_client.get_object(Bucket=bucket, Key=file_key)["Body"]

    return filebuffer


def get_singleview_data(image_file, image_transform, device, image_over_white=None):
    """
    Helper to prepare input data for single view conditional generation.

    Arguments:
        image_file:         A str or file object.
        image_transform:    A function that takes a PIL.Image and returns a torch.tensor.
        device:             A torch device.
        image_over_white:   (optional) Pass True to overlay the input image over a white background
                            when removing the alpha channel. Pass False to overlay over a black
                            background. Pass None to autodetect based on the given image's mode -
                            greyscale images are overlaid on white and RGB images are overlaid on
                            black.
    """

    if isinstance(image_file, str):
        file_base_name = osp.basename(image_file).split(".")[0]
        if image_file.startswith("s3://"):
            image_file = get_s3_object(image_file)
    else:
        file_base_name = uuid.uuid4()

    image = load_image(image_file, image_over_white=image_over_white)
    image = image_transform(image)

    images = image.to(device).unsqueeze(0).float()
    img_idx = torch.from_numpy(np.array([0])).long().to(device)

    data = {}
    data["images"] = images
    data["img_idx"] = img_idx
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data


def get_multiview_data(image_files, views, image_transform, device):

    image_files = [
        (
            get_s3_object(image_file)
            if isinstance(image_file, str) and image_file.startswith("s3://")
            else image_file
        )
        for image_file in image_files
    ]

    file_base_name = uuid.uuid4()

    images = [
        image_transform(load_image(image_file)).to(device).unsqueeze(0).float()
        for image_file in image_files
    ]
    images = torch.cat(images, dim=0).unsqueeze(0)
    img_idx = torch.from_numpy(np.array(views)).long().unsqueeze(0).to(device)

    data = {}
    data["images"] = images
    data["img_idx"] = img_idx
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data


def get_voxel_data_json(voxel_file, voxel_resolution, device):
    if isinstance(voxel_file, str):
        file_base_name = osp.basename(voxel_file).split(".")[0]
    else:
        file_base_name = uuid.uuid4()

    with open(voxel_file, "r") as f:
        voxel_json = json.load(f)

    voxel_resolution = voxel_json["resolution"]
    occupancy_arr = np.array(voxel_json["occupancy"])
    color_numpy = np.array(voxel_json["color"])

    voxel_grid = make_a_grid(
        occupancy_arr,
        color_numpy,
        voxel_resolution=voxel_resolution,
    )[:, :, :, 3:4]

    voxel_grid = voxel_grid.transpose(3, 0, 1, 2).astype(np.float32)
    voxel_grid = torch.from_numpy(voxel_grid).to(device)
    voxels = voxel_grid.unsqueeze(0).float()

    data = {}
    data["voxels"] = voxels
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data


def get_image_transform(args, train_step=False):

    transform = []

    if (
        hasattr(args, "use_image_augmentation")
        and args.use_image_augmentation
        and train_step
    ):
        transform = transform + [
            RandomAffine(
                degrees=args.image_rotation_range,
                translate=args.image_translation_range,
                scale=args.image_scale_range,
            ),
            ColorJitter(
                brightness=args.image_brightness_range,
                contrast=args.image_contrast_range,
                saturation=args.image_saturation_range,
                hue=args.image_hue,
            ),
        ]

    transform = transform + [
        Resize(args.n_px, interpolation=Image.BICUBIC),
        CenterCrop(args.n_px),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]

    transform = Compose(transform)

    return transform


def get_image_transform_latent_model():
    transform = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    return transform


def get_pointcloud_data(pointcloud_file, device):
    if isinstance(pointcloud_file, str):
        file_base_name = osp.basename(pointcloud_file).split(".")[0]
    else:
        file_base_name = uuid.uuid4()

    loaded_pointcloud = load_arr_file(
        file=pointcloud_file, keys=["points"], file_type="h5df"
    )

    pointcloud = loaded_pointcloud["points"].astype(np.float32)
    pointcloud = torch.from_numpy(pointcloud).to(device)
    pointcloud = pointcloud.unsqueeze(0).float()

    data = {}
    data["Pointcloud"] = pointcloud
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data


def get_mv_dm_data(image_files, views, image_transform, device):

    image_files = [
        (
            get_s3_object(image_file)
            if isinstance(image_file, str) and image_file.startswith("s3://")
            else image_file
        )
        for image_file in image_files
    ]

    file_base_name = uuid.uuid4()

    images = [
        image_transform(load_image(image_file)).to(device).unsqueeze(0).float()
        for image_file in image_files
    ]
    images = torch.cat(images, dim=0).unsqueeze(0)
    img_idx = torch.from_numpy(np.array(views)).long().unsqueeze(0).to(device)

    data = {}
    data["depth"] = images
    data["img_idx"] = img_idx
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data


def get_sketch_data(image_file, image_transform, device, image_over_white=True):
    
    if isinstance(image_file, str):
        file_base_name = osp.basename(image_file).split(".")[0]
        if image_file.startswith("s3://"):
            image_file = get_s3_object(image_file)
    else:
        file_base_name = uuid.uuid4()

    image = load_image(image_file, image_over_white=image_over_white)
    image = image_transform(image)

    images = image.to(device).unsqueeze(0).float()
    img_idx = torch.from_numpy(np.array([0])).long().to(device)

    data = {}
    data["images"] = images
    data["img_idx"] = img_idx
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data

def get_sv_dm_data(image_file, image_transform, device, image_over_white=None):
    
    if isinstance(image_file, str):
        file_base_name = osp.basename(image_file).split(".")[0]
        if image_file.startswith("s3://"):
            image_file = get_s3_object(image_file)
    else:
        file_base_name = uuid.uuid4()

    image = load_image(image_file, image_over_white=image_over_white)
    image = image_transform(image)

    images = image.to(device).unsqueeze(0).float()
    img_idx = torch.from_numpy(np.array([0])).long().to(device)

    data = {}
    data["depth"] = images
    data["img_idx"] = img_idx
    data["low"] = torch.zeros((1, 1, 46, 46, 46)).to(device)
    data["id"] = [file_base_name]

    return data