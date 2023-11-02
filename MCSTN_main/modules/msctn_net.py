
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.heads.RPN_head import RPN

from mmcv.ops import Voxelization
from modules.Voxel_encoder.Dynamic_voxel import DynamicVFE

from modules.SST_2.no_window_4_update_loss import Direct_attenttion10

class MSCTN_Tracking(nn.Module):
    def __init__(self, opts):
        super(MSCTN_Tracking, self).__init__()
        self.opts = opts
        # DV
        self.voxel_layer = Voxelization(voxel_size=[0.3, 0.3, 4.8], max_voxels=(-1, -1), max_num_points=-1,
                                        point_cloud_range=[-4.8, -4.8, -2.4, 4.8, 4.8, 2.4])
        self.voxel_encoder = DynamicVFE(in_channels=3, feat_channels=[64, 128], with_distance=False,
                                        voxel_size=(0.3, 0.3, 4.8), with_cluster_center=True, with_voxel_center=True,
                                        point_cloud_range=(-4.8, -4.8, -2.4, 4.8, 4.8, 2.4))
        self.RPN = RPN()

        self.backbone10 = Direct_attenttion10(d_model=[128, ] * 2, nhead=[8, ] * 2,
                                              num_blocks=2, dim_feedforward=[256, ] * 2,
                                              dropout=0.0, output_shape=[32, 32],
                                              activation="gelu", conv_in_channel=128,
                                              conv_kwargs=[
                                                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                                                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                                                dict(kernel_size=3, dilation=2, padding=2, stride=1),
                                              ],
                                              conv_out_channel=128, num_attached_conv=3)
        self.train_test = opts.train_test

    def forward(self, template, search):
        r"""
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        bs = template.shape[0]
        template_points = []
        search_points = []
        for i in range(bs):
            template_points.append(template[i])
            search_points.append(search[i])

        # 2.dynamic voxelization
        search_voxels, search_coors = self.voxelize(search_points)
        template_voxels, template_coors = self.voxelize(template_points)

        search_voxel_features, search_voxels_coors = self.voxel_encoder(search_voxels, search_coors)
        template_voxel_features, template_voxels_coors = self.voxel_encoder(template_voxels, template_coors)

        out_block_bev = self.backbone10(search_voxel_features, search_voxels_coors, template_voxel_features,
                                        template_voxels_coors, train_test=self.train_test)
        train_test = self.train_test

        pred_hm = []
        pred_loc = []
        pred_z_axis = []
        if train_test=='train':
            for i in range(len(out_block_bev)):
                pred_hm_i, pred_loc_i, pred_z_axis_i = self.RPN(out_block_bev[i])
                pred_hm.append(pred_hm_i)
                pred_loc.append(pred_loc_i)
                pred_z_axis.append(pred_z_axis_i)
        elif train_test == 'test':
            pred_hm_i, pred_loc_i, pred_z_axis_i = self.RPN(out_block_bev[len(out_block_bev)-1])
            pred_hm.append(pred_hm_i)
            pred_loc.append(pred_loc_i)
            pred_z_axis.append(pred_z_axis_i)

        return pred_hm, pred_loc, pred_z_axis

    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def hard_voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer2(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch



