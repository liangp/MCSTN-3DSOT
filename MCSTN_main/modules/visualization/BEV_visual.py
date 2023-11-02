import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def get_theta(box3d_corner):
    theat = (np.arctan2(box3d_corner[:, 5, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 5, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 5, 0] - box3d_corner[:, 6, 0],
                        box3d_corner[:, 6, 1] - box3d_corner[:, 5, 1]) +
             np.arctan2(box3d_corner[:, 6, 1] - box3d_corner[:, 2, 1],
                        box3d_corner[:, 6, 0] - box3d_corner[:, 2, 0]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 2, 0],
                        box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return theat


def distanceBB_Gaussian(box1, box2, sigma=1):
    # box1: 候选框, 1, 8, 3
    # box2: gt, 1, 7
    gt_boxes_center = np.stack([box2[:, 0], box2[:, 1]], axis=1).squeeze()  # [2, ]
    pred_boxes_center = np.mean(box1[:, :, :2], axis=1).squeeze()    # [2, ]

    off1 = np.array([
        pred_boxes_center[0], pred_boxes_center[1],
        get_theta(box2)
    ])
    off2 = np.array([
        gt_boxes_center.center[0], gt_boxes_center.center[1],
        np.array(box2[0][6])/np.pi * 180
    ])
    dist = np.linalg.norm(off1 - off2)
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return score


def computer_3d_box_cam2(x,y,z,l,h,w,yaw):
    """
    Return :3 * n in cam2 coordinate
    """
    R=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    x_corners=[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners=[0,0,0,0,-h,-h,-h,-h]
    z_cornes=[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2=np.dot(R,np.vstack([x_corners,y_corners,z_cornes]))
    corners_3d_cam2+=np.vstack([x,y,z])
    return corners_3d_cam2



def draw_box_2D(pyplot_axis, vertices, axes=[0, 1], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    # [[74 113  74 113]
    #  [40  40  73  73]]
    # print("*******************")
    # print("vertices:",vertices)
    connections = [[0, 1], [1, 2], [2, 3], [3, 0]]  # 顺序进行调整
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=1)

def draw_box_3D(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

def visualize_feature_maps_crop(X,  axes=[2, 0], save_filename=None,
                                velo_corners_3d=[],
                                candidate_BBs=[],
                                search_velo_corners_3d=[],
                                pred_boxes = [],
                                max_bboxs=[],
                                max_bboxs_1=[]):
    nc = np.ceil(np.sqrt(X.shape[0]))  # column
    nr = np.ceil(X.shape[0] / nc)  # row
    nc = int(nc)
    nr = int(nr)
    plt.figure(figsize=(64,64))
    # for i in range(X.shape[0]):
    for i in range(64):
        ax = plt.subplot(nr, nc, i + 1)
        # ax.imshow(X[i, :, :].detach().cpu().numpy(), vmin=0, vmax=20, cmap="jet")
        ax.imshow(X[i, :, :].detach().cpu().numpy(), cmap="jet")
        # plt.colorbar()
        for velo_corner_3d in velo_corners_3d:
            draw_box_3D(ax, velo_corner_3d, axes=[0, 1], color="r")  # gt model

        for search_velo_corner_3d in search_velo_corners_3d:
            draw_box_3D(ax, search_velo_corner_3d, axes=[0, 1], color="r")  # gt search

        # 这个地方是画候选框
        for candidate_BB in candidate_BBs:
            draw_box_3D(ax,candidate_BB,axes=[0,1],color="r")  # search产生的候选

        for box in pred_boxes:
            draw_box_3D(ax,box,axes=[0,1],color="b")  # search产生的候选

        for max_box in max_bboxs:
            draw_box_2D(ax, max_box, axes=[0, 1], color="y")

        for max_box_1 in max_bboxs_1:
            draw_box_2D(ax, max_box_1, axes=[0, 1], color="y")

        # ax.axis("off")
    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.show()
    plt.close()


def visualize_feature_maps_crop_lidar(X, axes=[0, 1], save_filename=None, gt_bboxs=[], cand_bboxs=[]):
    nc = np.ceil(np.sqrt(X.shape[0]))  # column
    nr = np.ceil(X.shape[0] / nc)  # row
    nc = int(nc)
    nr = int(nr)
    plt.figure(figsize=(64, 64))
    # for i in range(X.shape[0]):
    for i in range(64):  # 可视化前5个
        ax = plt.subplot(nr, nc, i + 1)
        # ax.imshow(X[i, :, ].detach().cpu().numpy(), vmin=0, vmax=20, cmap="jet")
        ax.imshow(X[i, :, ].detach().cpu().numpy(), cmap="jet")

        for gt_box in gt_bboxs:
            draw_box_3D(ax, gt_box, axes=[0, 1], color="r")  # gt model

        for cand_bbox in cand_bboxs:
            draw_box_3D(ax, cand_bbox, axes=[0, 1], color="g")

        # ax.axis("off")
    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.show()
    plt.close()

