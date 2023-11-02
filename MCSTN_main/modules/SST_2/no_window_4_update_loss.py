# 动态体素编码后，不划分窗口直接进行self attention和cross attention
# template DansNet相加，search在cross后DansNet相加
import torch
from torch import nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from modules.ops.linear_attention import LinearAttention


class self_attention(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(self_attention, self).__init__()

        self.dim = d_model // nhead  # 128/8 = 16
        self.nhead = nhead  # 8

        # position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, search_feat, search_coors, search_mask):
        '''

        Parameters
        ----------
        search_feat [B, 512, C]
        search_coors [B, 512, 3]
        search_mask  [B, 512]  0表示padding的
        Returns
        -------

        '''
        bs = search_feat.size(0)
        search_xyz = torch.as_tensor(search_coors, dtype=search_feat.dtype)

        search_feat_pos = search_feat + self.pos_mlp(search_xyz)
        # multi-head attention
        query = self.q_proj(search_feat_pos).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(search_feat_pos).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(search_feat_pos).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=search_mask, kv_mask=search_mask)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([search_feat, message], dim=2))
        message = self.norm2(message)

        new_search_feat = search_feat + message  # 得到融合template后的search特征 [B,512,C]

        return new_search_feat


class cross_attention(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(cross_attention, self).__init__()

        self.dim = d_model // nhead  # 128/8 = 16
        self.nhead = nhead  # 8

        # position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, search_feat, search_coors, search_mask, template_feat, template_coors, template_mask):
        '''

        Parameters
        ----------
        search_feat [B, 512, C]
        search_coors [B, 512, 3]
        search_mask  [B, 512]  0表示padding的
        Returns
        -------

        '''
        bs = search_feat.size(0)

        search_xyz = torch.as_tensor(search_coors, dtype=template_feat.dtype)


        search_feat_pos = search_feat + self.pos_mlp(search_xyz)

        query = self.q_proj(search_feat_pos).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(template_feat).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(template_feat).view(bs, -1, self.nhead, self.dim)
        # template->search
        message = self.attention(query, key, value, q_mask=search_mask, kv_mask=template_mask)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([search_feat, message], dim=2))
        message = self.norm2(message)

        new_search_feat = search_feat + message

        return new_search_feat


class no_window_block(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(no_window_block, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead

        self.self_attention1 = self_attention(d_model, nhead)
        self.cross_attention = cross_attention(d_model, nhead)

    def forward(self, search_feat, search_xyz, search_mask, template_feat, template_xyz, template_mask, search_original_indx_list, num_search, template_original_indx_list, num_template):
        dim = search_feat.shape[-1]
        self_search = self.self_attention1(search_feat, search_xyz, search_mask)
        self_template = self.self_attention1(template_feat, template_xyz, template_mask)

        cross_search = self.cross_attention(self_search, search_xyz, search_mask, self_template, template_xyz, template_mask)

        search_original_indx = torch.cat(search_original_indx_list, dim=0)
        search_mask = search_mask.bool()
        search_feats = cross_search[search_mask]
        all_search_voxels = torch.zeros([num_search, dim], dtype=search_feat.dtype, device=search_feat.device)
        all_search_voxels[search_original_indx] = search_feats  # [N,C]

        template_original_indx = torch.cat(template_original_indx_list, dim=0)
        template_mask = template_mask.bool()
        template_feats = self_template[template_mask]
        all_template_voxels = torch.zeros([num_template, dim], dtype=template_feat.dtype, device=template_feat.device)
        all_template_voxels[template_original_indx] = template_feats

        return all_search_voxels, all_template_voxels


class Direct_attenttion10(nn.Module):

    def __init__(
            self,
            d_model=[],  #
            nhead=[],  #
            num_blocks=1,
            dim_feedforward=[],
            dropout=0.0,
            activation="gelu",
            output_shape=[32, 32],
            num_attached_conv=3,
            conv_in_channel=64,
            conv_out_channel=64,
            norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False),
            debug=True,
            in_channel=None,
            conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
            checkpoint_blocks=[],
            conv_shortcut=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        self.conv_shortcut = conv_shortcut
        self.num_blocks = num_blocks

        block_list_search = []
        for i in range(num_blocks):
            block_list_search.append(no_window_block(d_model=d_model[i], nhead=nhead[i])
                                     )

        self.block_list_search = nn.ModuleList(block_list_search)

        self._reset_parameters()
        self.output_shape = output_shape

        self.debug = debug

        self.num_attached_conv = num_attached_conv
        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel  # 64
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.BatchNorm2d(conv_out_channel),
                        nn.ReLU(inplace=True)
                    )
                conv_list.append(convnormrelu)
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, search_feat0, search_coors, template_feat0, template_coors, train_test):
        '''

        Parameters
        ----------
        search_feat N1,C
        search_coors N1,4
        template_feat N2,C
        template_coors N2,4

        Returns
        -------

        '''
        num_search = search_feat0.shape[0]
        num_template = template_feat0.shape[0]
        dim = search_feat0.shape[1]
        batch_size = search_coors[:, 0].max().item() + 1
        search_feat, search_xyz, search_mask, template_feat, template_xyz, template_mask, search_original_indx_list, \
        template_original_indx_list = self.padding_tensor(search_feat0, search_coors, template_feat0, template_coors)

        output_template_list = []
        output_template_list.append(template_feat0)
        output_search_list = []
        output_search_list.append(search_feat0)
        paddng_search = []
        for i, block in enumerate(self.block_list_search):
            self_cross_2d_search, self_2d_template = block(search_feat, search_xyz, search_mask, template_feat, template_xyz,
                                    template_mask, search_original_indx_list, num_search, template_original_indx_list, num_template)
            output_search_list.append(self_cross_2d_search)
            output_template_list.append(self_2d_template)
            search_feat = sum(output_search_list)  # N,C
            search_feat = self.single_padding_tensor(search_feat, search_coors)
            paddng_search.append(search_feat)

        assert len(output_search_list) == self.num_blocks + 1
        ouput_feat_loss = []
        if train_test == 'train':
            for i in range(len(paddng_search)):
                search_feat = paddng_search[i]
                search_original_indx = torch.cat(search_original_indx_list, dim=0)
                search_mask = search_mask.bool()
                search_feats = search_feat[search_mask]
                all_search_voxels = torch.zeros([num_search, dim], dtype=search_feat.dtype, device=search_feat.device)
                all_search_voxels[search_original_indx] = search_feats  # [N,C]
                output_search = self.recover_bev(all_search_voxels, search_coors, batch_size)[0]

                results = []
                results.append(output_search)
                if self.num_attached_conv > 0:
                    for conv in self.conv_layer:
                        temp = conv(output_search)
                        results.append(temp)
                        if temp.shape == output_search.shape and self.conv_shortcut:
                            output_search = sum(results)
                        else:
                            output_search = temp
                ouput_feat_loss.append(output_search)
            assert len(ouput_feat_loss) == self.num_blocks

        elif train_test == 'test':
            search_original_indx = torch.cat(search_original_indx_list, dim=0)
            search_mask = search_mask.bool()
            search_feats = search_feat[search_mask]
            all_search_voxels = torch.zeros([num_search, dim], dtype=search_feat.dtype, device=search_feat.device)
            all_search_voxels[search_original_indx] = search_feats  # [N,C]
            output_search = self.recover_bev(all_search_voxels, search_coors, batch_size)[0]
            results = []
            results.append(output_search)
            if self.num_attached_conv > 0:
                for conv in self.conv_layer:
                    temp = conv(output_search)
                    results.append(temp)
                    if temp.shape == output_search.shape and self.conv_shortcut:
                        output_search = sum(results)
                    else:
                        output_search = temp
            ouput_feat_loss.append(output_search)


        return ouput_feat_loss

    def padding_tensor(self, search_feat, search_coors, template_feat, template_coors):
        dim = search_feat.shape[-1]
        batch_size = search_coors[:, 0].max().item() + 1
        padding_search_feat = torch.zeros([batch_size, 512, dim], dtype=search_feat.dtype, device=search_feat.device)
        padding_search_coors = torch.zeros([batch_size, 512, 3], dtype=search_coors.dtype, device=search_coors.device)
        padding_search_mask = torch.zeros([batch_size, 512], dtype=search_coors.dtype, device=search_coors.device)

        padding_template_feat = torch.zeros([batch_size, 256, dim], dtype=template_feat.dtype, device=template_feat.device)
        padding_template_coors = torch.zeros([batch_size, 256, 3], dtype=template_coors.dtype, device=template_coors.device)
        padding_template_mask = torch.zeros([batch_size, 256], dtype=template_coors.dtype, device=template_coors.device)

        search_original_indx_list = []
        template_original_indx_list = []

        for batch_itt in range(batch_size):

            batch_search_mask = search_coors[:, 0] == batch_itt
            batch_search_feats = search_feat[batch_search_mask]  # 找出当前batch的特征
            length = batch_search_feats.shape[0]
            padding_search_feat[batch_itt][:length, :] = batch_search_feats
            padding_search_coors[batch_itt][:length, :] = search_coors[batch_search_mask][:length, 1:4]
            padding_search_mask[batch_itt][:length] = 1

            batch_template_mask = template_coors[:, 0] == batch_itt
            batch_template_feats = template_feat[batch_template_mask]  # 找出当前batch的特征
            length2 = batch_template_feats.shape[0]
            padding_template_feat[batch_itt][:length2, :] = batch_template_feats
            padding_template_coors[batch_itt][:length2, :] = template_coors[batch_template_mask][:length2, 1:4]
            padding_template_mask[batch_itt][:length2] = 1

            search_original_indx_list.append(torch.where(batch_search_mask)[0])  # 体素对应的初始位置
            template_original_indx_list.append(torch.where(batch_template_mask)[0])

        return padding_search_feat, padding_search_coors, padding_search_mask, \
               padding_template_feat, padding_template_coors, padding_template_mask, \
               search_original_indx_list, template_original_indx_list

    def single_padding_tensor(self, search_feat, search_coors):
        dim = search_feat.shape[-1]
        batch_size = search_coors[:, 0].max().item() + 1
        padding_search_feat = torch.zeros([batch_size, 512, dim], dtype=search_feat.dtype, device=search_feat.device)
        for batch_itt in range(batch_size):
            batch_search_mask = search_coors[:, 0] == batch_itt
            batch_search_feats = search_feat[batch_search_mask]  # 找出当前batch的特征
            length = batch_search_feats.shape[0]
            padding_search_feat[batch_itt][:length, :] = batch_search_feats

        return padding_search_feat

    def single_padding_tensor_template(self, template_feat, template_coors):
        dim = template_feat.shape[-1]
        batch_size = template_coors[:, 0].max().item() + 1
        padding_template_feat = torch.zeros([batch_size, 256, dim], dtype=template_feat.dtype, device=template_feat.device)
        for batch_itt in range(batch_size):
            batch_search_mask = template_coors[:, 0] == batch_itt
            batch_search_feats = template_feat[batch_search_mask]  # 找出当前batch的特征
            length = batch_search_feats.shape[0]
            padding_template_feat[batch_itt][:length, :] = batch_search_feats

        return padding_template_feat

    def list2d_to_bev(self, feats_2d_list, original_indx_list, coors):
        pass

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]   B C H W
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        padding_masks = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)
            padding_mask = -torch.ones(nx * ny, dtype=torch.int, device=voxel_feat.device)

            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :]
            voxels = voxels.t()

            canvas[:, indices] = voxels
            padding_mask[indices] = 0

            batch_canvas.append(canvas)
            padding_masks.append(padding_mask)

        batch_canvas = torch.stack(batch_canvas, 0)
        padding_masks = torch.stack(padding_masks, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)
        padding_masks = padding_masks.view(batch_size, nx*ny) == -1

        return batch_canvas, padding_masks
