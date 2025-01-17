import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 64, 128], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128*3, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(32, [0.8], [32], in_channel=512, mlp_list=[[256, 512, 1024]])
        self.skip_conet0 = PCT(6, 128)
        self.skip_conet1 = PCT(128*3, 128*3)
        self.skip_conet2 = PCT(512, 512)
        self.skip_conet3 = PCT(1024, 1024)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128*3, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l3_points = self.skip_conet3(l3_points, cls_label)
        l2_points = self.fp3(l2_xyz, l3_xyz, self.skip_conet2(l2_points, cls_label), l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, self.skip_conet1(l1_points, cls_label), l2_points)
        # cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, self.skip_conet0(l0_points, cls_label), l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss


class PCT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PCT, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(in_channels=128, out_channels=128, d_tran = 128)
        self.sa2 = SA_Layer(in_channels=128, out_channels=128, d_tran = 128)
        self.sa3 = SA_Layer(in_channels=128, out_channels=128, d_tran = 128)
        self.sa4 = SA_Layer(in_channels=128, out_channels=128, d_tran = 128)
        self.sa5 = SA_Layer(in_channels=128, out_channels=128, d_tran = 128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(128*5, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, out_channel, 1)
        # self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = self.sa5(x4)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = torch.cat((x1), dim=1)
        # x = x1
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.concat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1024 + 64
        x = torch.concat((x, x_global_feature), 1)  # 1024 * 3 + 64
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        # x = self.convs3(x)
        # x = F.log_softmax(x, dim=1)

        return x


class SA_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, d_tran = 128):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(in_channels, d_tran, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, d_tran, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(in_channels, d_tran, 1)
        self.trans_conv = nn.Conv1d(d_tran, out_channels, 1)
        self.after_norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x