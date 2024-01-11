import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SPQ_SubNet import SPQSubNet
from model.TPQ_SubNet import TPQSubNet


class STSURNet(nn.Module):
    def __init__(self, patch_per_frame=64, key_frame_nb=12):
        super(STSURNet, self).__init__()
        self.patch_per_frame = patch_per_frame
        self.key_frame_nb = key_frame_nb
        self.patches = self.patch_per_frame * self.key_frame_nb

        # 子网
        self.spqSubNet = SPQSubNet()
        self.tpqSubNet = TPQSubNet()

        # source 与 compress 使用的FC
        self.sc_fc1 = nn.Linear(512 * 3, 512)
        self.sc_fc2 = nn.Linear(512, 1)
        self.sc_fc3 = nn.Linear(512 * 3, 1)

        # optical 光流使用的FC
        self.o_fc1 = nn.Linear(512 * 3, 512)
        self.o_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # input X (s, c, so, co)
        hs = self.spqSubNet(x[0].float())
        hc = self.spqSubNet(x[1].float())
        hso = self.tpqSubNet(x[2].float())
        hco = self.tpqSubNet(x[3].float())

        # 连接空间特征，全连接
        hsc = torch.cat((hc, hs, hc - hs), dim=1)
        hsc = hsc.view(self.patches, 512 * 3)
        spq = F.dropout(F.relu(self.sc_fc1(hsc)), p=0.5)
        spq = self.sc_fc2(spq)

        # 连接时间特征，全连接
        ht = torch.cat((hco, hso, hco - hso), dim=1)
        ht = ht.view(self.patches, 512 * 3)
        tpq = F.dropout(F.relu(self.o_fc1(ht)), p=0.5)
        tpq = self.o_fc2(tpq)

        patch_q = torch.mul(spq, tpq)  # 每个patch的时空特征融合，点乘（这里我感觉点乘不了，论文写的点乘，但是我觉得他想说对应元素相乘）

        # 准备权重
        wei = self.sc_fc3(hsc)
        wei = wei-torch.min(wei)
        wei_flat = wei.view(self.key_frame_nb, self.patch_per_frame)
        norm = wei_flat.sum(dim=1, keepdim=True) + 1e-10
        normalized_wei = wei_flat/norm

        # 加权求和
        frame_sur = torch.sum(torch.mul(patch_q.reshape(self.key_frame_nb, self.patch_per_frame), normalized_wei),
                              dim=1)  # 评分与权重对应元素相乘，再按行求和（每行是一帧的所有patch）
        outputs = torch.mean(frame_sur)
        return outputs
