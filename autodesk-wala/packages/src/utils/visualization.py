import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append("../")

import trimesh
import torch
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from skimage.measure import block_reduce

import numpy as np


def save_point_cloud_as_ply(filename, points, colors=None):
    """
    Save a point cloud to a PLY file.

    :param filename: str
        The output filename for the PLY file.
    :param points: ndarray
        A Nx3 array of points.
    :param colors: ndarray, optional
        A Nx3 array of colors, with each row representing the color (in RGB format)
        for each point. Each color component can be a value between 0 and 255.
    """
    if not filename.lower().endswith(".ply"):
        raise ValueError("Filename must end with '.ply'")

    with open(filename, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(points)))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        if colors is not None:
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for i in range(len(points)):
            point_str = "{} {} {}".format(points[i, 0], points[i, 1], points[i, 2])
            if colors is not None:
                color_str = " {} {} {}".format(colors[i, 0], colors[i, 1], colors[i, 2])
                ply_file.write(point_str + color_str + "\n")
            else:
                ply_file.write(point_str + "\n")


def save_voxel_image(voxel_data, file_name):
    voxel_data = voxel_data.transpose(2, 0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxel_data, edgecolor="k")
    ax.set_title("Voxel")
    plt.savefig(file_name)


def save_voxel_and_image(voxel_list, image, file_name):
    fig = plt.figure(figsize=(40, 2.5))

    image = np.array(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    ax2 = fig.add_subplot(121)
    ax2.imshow(image, cmap="gray")
    ax2.set_title("Image")

    for i, voxel in enumerate(voxel_list):
        ax = fig.add_subplot(1, voxel_list.shape[0] + 1, i + 2, projection="3d")
        ax.voxels(voxel, edgecolor="k")
        ax.set_title("Voxel {}".format(i + 1))

    plt.savefig(file_name)


def save_normalized_image_matplotlib(img, file_name):
    img = np.array(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    plt.imsave(file_name, img, cmap="gray", vmin=0, vmax=1)


def save_mesh(
    voxels, out_file=None, threshold=0.5, smooth=True, box_size=1.1, lamb=0.05
):
    from utils import libmcubes

    n_x, n_y, n_z = voxels.shape
    box_size = box_size

    threshold = np.log(threshold) - np.log(1.0 - threshold)
    # Make sure that mesh is watertight

    # voxels = np.pad(voxels, 1, 'constant', constant_values=-1e6)

    vertices, triangles = libmcubes.marching_cubes(voxels, threshold)
    vertices -= 0.5
    # Undo padding
    # vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    normals = None
    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals, process=False)
    # print(mesh)
    if smooth == True:
        try:
            mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=lamb)
        except:
            print("Error Smoothing mesh")

    if out_file is not None:
        mesh.export(out_file)
    else:
        return mesh


def save_mesh_png(voxels, out_file, threshold=0.5):
    from utils import libmcubes

    n_x, n_y, n_z = voxels.shape
    box_size = 1.1

    threshold = np.log(threshold) - np.log(1.0 - threshold)
    # Make sure that mesh is watertight

    # voxels = np.pad(voxels, 1, 'constant', constant_values=-1e6)

    vertices, triangles = libmcubes.marching_cubes(voxels, threshold)
    vertices -= 0.5
    # Undo padding
    # vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    normals = None
    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals, process=False)
    # print(mesh)
    try:
        mesh = trimesh.smoothing.filter_laplacian(mesh)
    except:
        print("Error Smoothing mesh")

    mesh.visual.material = trimesh.visual.material.SimpleMaterial(
        diffuse=None, ambient=None, specular=None, glossiness=4
    )

    # scene will have automatically generated camera and lights
    scene = mesh.scene()
    rotate = trimesh.transformations.rotation_matrix(
        angle=np.radians(70.0), direction=[-1, 1, 0], point=scene.centroid
    )
    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [640, 480]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = 120 * (scene.camera.resolution / scene.camera.resolution.max())

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()
    # print(pixels.shape)
    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(rotate, camera_old)

    # apply the new transform
    scene.graph[scene.camera.name] = camera_new
    # print(scene)
    png = scene.save_image(resolution=[1920, 1080], visible=False)

    with open(out_file, "wb") as f:
        f.write(png)
        f.close()


def save_point_cloud(voxels, out_file):

    result = np.nonzero(voxels)
    pc = list(zip(result[0], result[1], result[2]))

    file1 = open(out_file, "w")
    file1.write("ply\n")
    file1.write("format ascii 1.0\n")
    file1.write("element vertex " + str(len(pc)) + " \n")
    file1.write("property float x\n")
    file1.write("property float y\n")
    file1.write("property float z\n")
    file1.write("end_header\n")
    for index in range(len(pc)):
        file1.write(
            str(pc[index][1]) + " " + str(pc[index][2]) + " " + str(pc[index][0]) + "\n"
        )
    file1.close()


def make_3d_grid(bb_min, bb_max, shape):
    """Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def visualize_voxels(voxels, out_file=None, show=False, transpose=True):
    r"""Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    """
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if transpose == True:
        voxels = voxels.transpose(2, 0, 1)
    # else:
    # voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor="k")
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.axis("off")
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_voxels_texture(voxels, out_file=None, show=False, transpose=True):

    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if transpose == True:
        voxels = voxels.transpose(2, 0, 1, 3)
    # else:
    # voxels = voxels.transpose(2, 0, 1)
    # ax.voxels(voxels, edgecolor='k')
    ax.voxels(voxels[:, :, :, 3], facecolors=voxels[:, :, :, 0:3])
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.axis("off")
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def sketch_point_cloud(points, save_loc=None, lmit=0.4):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # plt.axis('off')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-1 * lmit, lmit])
    ax.set_ylim([-1 * lmit, lmit])
    ax.set_zlim([-1 * lmit, lmit])
    ax.view_init(30, 0)
    if save_loc != None:
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis("off")
        plt.savefig(save_loc)

    plt.show()


def visualize_pointcloud(points, save_loc=None, show=False):
    r"""Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    """
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if save_loc is not None:
        plt.axis("off")
        plt.savefig(save_loc)
    else:
        plt.show()
    plt.close(fig)


def plot_real_pred(real_points, pred_points, num_plots, lmit=0.6, save_loc=None):
    fig = plt.figure(figsize=(40, 20))
    for i in range(num_plots):
        plt_num = str(i + 1) + "21"
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection="3d")
        real_point = real_points[i]
        ax.scatter(real_point[:, 2], real_point[:, 0], real_point[:, 1])
        # ax.view_init(-30, 45)
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        ax.grid(False)

        plt_num = str(i + 1) + "22"
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection="3d")
        pred_point = pred_points[i]
        ax.scatter(pred_point[:, 2], pred_point[:, 0], pred_point[:, 1])
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        # ax.view_init(-30, 45)
        ax.grid(False)
    # plt.axis('off')
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return

    plt.show()


def plot_real_inter_pred(
    real_points, inter_points, pred_points, num_plots, lmit=0.6, save_loc=None
):
    fig = plt.figure(figsize=(40, 20))
    for i in range(num_plots):
        plt_num = str(i + 1) + "31"
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection="3d")
        real_point = real_points[i]
        ax.scatter(real_point[:, 2], real_point[:, 0], real_point[:, 1])
        # ax.view_init(-30, 45)
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        ax.grid(False)

        plt_num = str(i + 1) + "32"
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection="3d")
        inter_point = inter_points[i]
        ax.scatter(inter_point[:, 2], inter_point[:, 0], inter_point[:, 1])
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        # ax.view_init(-30, 45)
        ax.grid(False)

        plt_num = str(i + 1) + "33"
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection="3d")
        pred_point = pred_points[i]
        ax.scatter(pred_point[:, 2], pred_point[:, 0], pred_point[:, 1])
        ax.set_xlim([-1 * lmit, lmit])
        ax.set_ylim([-1 * lmit, lmit])
        ax.set_zlim([-1 * lmit, lmit])
        # ax.view_init(-30, 45)
        ax.grid(False)

    # plt.axis('off')
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return

    plt.show()


def multiple_plot_label_pc(batch_data_points, batch_labels, num_plots, save_loc=None):
    from sklearn.manifold import TSNE

    my_colors = {
        0: "orange",
        1: "red",
        2: "green",
        3: "blue",
        4: "grey",
        5: "gold",
        6: "violet",
        7: "pink",
        8: "navy",
        9: "black",
    }

    fig = plt.figure(figsize=(40, 20))

    for i in range(num_plots):
        plt_num = "1" + str(num_plots) + str(i + 1)
        # print(plt_num)
        ax = fig.gca(projection="3d")
        ax = fig.add_subplot(plt_num, projection="3d")
        data_points = batch_data_points[i]
        labels = batch_labels[i]
        for i, _ in enumerate(data_points):
            ax.scatter(
                data_points[i, 2],
                data_points[i, 0],
                data_points[i, 1],
                color=my_colors.get(labels[i], "black"),
            )
        ax.view_init(-30, 45)
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


def multiple_plot(batch_data_points, lmit=0.6, save_loc=None):

    my_colors = {
        0: "orange",
        1: "red",
        2: "green",
        3: "blue",
        4: "grey",
        5: "gold",
        6: "violet",
        7: "pink",
        8: "navy",
        9: "black",
    }

    fig = plt.figure(figsize=(40, 10))

    for i in range(len(batch_data_points)):
        plt_num = "1" + str(len(batch_data_points)) + str(i + 1)
        # print(plt_num)
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(plt_num, projection="3d")
        data_points = batch_data_points[i]

        for i, _ in enumerate(data_points):
            ax.scatter(
                data_points[i, 2], data_points[i, 0], data_points[i, 1], color="black"
            )

            ax.set_xlim([-1 * lmit, lmit])
            ax.set_ylim([-1 * lmit, lmit])
            ax.set_zlim([-1 * lmit, lmit])
            # ax.view_init(-30, 45)
            ax.grid(False)

            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        # ax.view_init(-30, 45)
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


def multiple_plot_voxel(batch_data_points, save_loc=None, transpose=True):

    fig = plt.figure(figsize=(40, 10))

    for i in range(len(batch_data_points)):
        plt_num = "1" + str(len(batch_data_points)) + str(i + 1)
        ax = fig.add_subplot(int(plt_num), projection=Axes3D.name)
        data_points = batch_data_points[i]

        if transpose == True:
            data_points = data_points.transpose(2, 0, 1)

        ax.voxels(data_points, edgecolor="k")
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        if transpose == False:
            ax.view_init(elev=-30, azim=45)
        else:
            ax.view_init(elev=30, azim=45)

    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


def multiple_plot_voxel_texture(batch_data_points, save_loc=None, transpose=True):

    fig = plt.figure(figsize=(40, 10))

    for i in range(len(batch_data_points)):
        plt_num = "1" + str(len(batch_data_points)) + str(i + 1)
        ax = fig.add_subplot(int(plt_num), projection=Axes3D.name)
        data_points = batch_data_points[i]

        if transpose == True:
            data_points = data_points.transpose(2, 0, 1, 3)

        # print(data_points[:,:,:, 3].shape, data_points[:,:,:, 0:3].shape, data_points[:,:,:, 0:3].max(), data_points[:,:,:, 0:3].min())
        ax.voxels(
            data_points[:, :, :, 3].astype(int), facecolors=data_points[:, :, :, 0:3]
        )
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        if transpose == False:
            ax.view_init(elev=-30, azim=45)
        else:
            ax.view_init(elev=30, azim=45)

    #         ax.set_xlabel('Y')
    #         ax.set_ylabel('Z')
    #         ax.set_zlabel('X')

    #         ax.view_init(elev=30, azim=-145)

    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


def sketch_labelled_pc(points, labels, save_loc=None, lmit=0.6):

    cmap = sns.cubehelix_palette(as_cmap=True)
    my_colors = {
        0: "orange",
        1: "red",
        2: "green",
        3: "blue",
        4: "grey",
        5: "gold",
        6: "violet",
        7: "pink",
        8: "navy",
        9: "black",
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i, data_point in enumerate(points):
        ax.scatter(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            color=my_colors.get(labels[i], "black"),
        )
    # ax.scatter(points[:,0], points[:,1],  points[:,2], c=labels[:], color = my_colors.get(labels[:], 'black')) #cmap=cmap
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-1 * lmit, lmit])
    ax.set_ylim([-1 * lmit, lmit])
    ax.set_zlim([-1 * lmit, lmit])
    ax.view_init(-30, 45)
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()

    plt.show()
    plt.close()


def plot_label_pc(data_points, labels, save_loc=None, lmit=0.6):

    my_colors = {
        0: "orange",
        1: "red",
        2: "green",
        3: "blue",
        4: "grey",
        5: "gold",
        6: "violet",
        7: "pink",
        8: "navy",
        9: "black",
    }

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax = fig.add_subplot(111, projection="3d")

    for i, _ in enumerate(data_points):
        ax.scatter(
            data_points[i, 0],
            data_points[i, 1],
            data_points[i, 2],
            color=my_colors.get(labels[i], "black"),
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-1 * lmit, lmit])
    ax.set_ylim([-1 * lmit, lmit])
    ax.set_zlim([-1 * lmit, lmit])
    ax.view_init(-30, 45)
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


def multiple_plot_label_pc(batch_data_points, batch_labels, num_plots, save_loc=None):
    from sklearn.manifold import TSNE

    my_colors = {
        0: "orange",
        1: "red",
        2: "green",
        3: "blue",
        4: "grey",
        5: "gold",
        6: "violet",
        7: "pink",
        8: "navy",
        9: "black",
    }

    fig = plt.figure(figsize=(60, 20))

    for i in range(num_plots):
        plt_num = "1" + str(num_plots) + str(i + 1)
        # print(plt_num)
        ax = fig.gca(projection="3d")
        ax = fig.add_subplot(plt_num, projection="3d")
        data_points = batch_data_points[i]
        labels = batch_labels[i]
        for i, _ in enumerate(data_points):
            ax.scatter(
                data_points[i, 0],
                data_points[i, 1],
                data_points[i, 2],
                color=my_colors.get(labels[i], "black"),
            )
        ax.view_init(-30, 45)
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


def visualize_voxels_texture(voxels, out_file=None, show=False, transpose=True):
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if transpose == True:
        voxels = voxels.transpose(0, 2, 1, 3)
    # else:
    # voxels = voxels.transpose(2, 0, 1)
    # ax.voxels(voxels, edgecolor='k')
    ax.voxels(voxels[:, :, :, 3], facecolors=voxels[:, :, :, 0:3])
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")

    # ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.axis("off")
        plt.savefig(out_file)
        plt.close(fig)
    if show:
        plt.show()
