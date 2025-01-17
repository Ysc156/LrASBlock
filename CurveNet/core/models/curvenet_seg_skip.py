import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *

curve_config = {
    'default': [[100, 5], [100, 5], None, None, None]
}


class CurveNet(nn.Module):
    def __init__(self, num_classes=50, category=16, k=32, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=2048, radius=0.2, k=k, in_channels=additional_channel, output_channels=64,
                         bottleneck_ratio=2, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=2048, radius=0.2, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4,
                         curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=512, radius=0.4, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2,
                         curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=512, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4,
                         curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=128, radius=0.8, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2,
                         curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=128, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4,
                         curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=32, radius=1.2, k=31, in_channels=256, output_channels=512, bottleneck_ratio=2,
                         curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=32, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4,
                         curve_config=curve_config[setting][3])

        self.cic51 = CIC(npoint=8, radius=2.0, k=7, in_channels=512, output_channels=1024, bottleneck_ratio=2,
                         curve_config=curve_config[setting][4])
        self.cic52 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024, output_channels=1024, bottleneck_ratio=4,
                         curve_config=curve_config[setting][4])
        self.cic53 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024, output_channels=1024, bottleneck_ratio=4,
                         curve_config=curve_config[setting][4])

        # decoder
        self.skip_connect4 = PCT(512, 512)
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512], att=[1024, 512, 256])
        self.up_cic5 = CIC(npoint=32, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4)

        self.skip_connect3 = PCT(256, 256)
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256], att=[512, 256, 128])
        self.up_cic4 = CIC(npoint=128, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)

        self.skip_connect2 = PCT(128, 128)
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128], att=[256, 128, 64])
        self.up_cic3 = CIC(npoint=512, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4)

        
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[64, 64], att=[128, 64, 32])
        self.up_cic2 = CIC(npoint=2048, radius=0.2, k=k, in_channels=128 + 64 + 64 + category + 3, output_channels=256,
                           bottleneck_ratio=4)
                           
        self.up_cic1 = CIC(npoint=2048, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)
        self.skip_connect0 = PCT(64, 64)
        self.fp0 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[64, 256])
        self.global_conv2 = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2))
        self.global_conv1 = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Conv1d(256, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(320, num_classes, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(256, 256 // 8, 1, bias=False),
                                nn.BatchNorm1d(256 // 8),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(256 // 8, 256, 1, bias=False),
                                nn.Sigmoid())

    def forward(self, xyz, l=None):
        batch_size = xyz.size(0)

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        l5_xyz, l5_points = self.cic51(l4_xyz, l4_points)
        l5_xyz, l5_points = self.cic52(l5_xyz, l5_points)
        l5_xyz, l5_points = self.cic53(l5_xyz, l5_points)

        # global features
        emb1 = self.global_conv1(l4_points)
        emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
        emb2 = self.global_conv2(l5_points)
        emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1

        # Feature Propagation layers

        # l4_points = self.fp4(l4_xyz, l5_xyz, l4_points, l5_points)
        l4_points = self.fp4(l4_xyz, l5_xyz, self.skip_connect4(l4_points), l5_points)
        l4_xyz, l4_points = self.up_cic5(l4_xyz, l4_points)

        # l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points = self.fp3(l3_xyz, l4_xyz, self.skip_connect3(l3_points), l4_points)
        l3_xyz, l3_points = self.up_cic4(l3_xyz, l3_points)

        # l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, self.skip_connect2(l2_points), l3_points)
        l2_xyz, l2_points = self.up_cic3(l2_xyz, l2_points)
        
        # l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points_e = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        if l is not None:
            l = l.view(batch_size, -1, 1)
            emb = torch.cat((emb1, emb2, l), dim=1)  # bs, 128 + 64 + 16, 1
        l = emb.expand(-1, -1, xyz.size(-1))
        x = torch.cat((l1_xyz, l1_points_e, l), dim=1)

        xyz, x = self.up_cic2(l1_xyz, x)
        xyz, x = self.up_cic1(xyz, x)
        #print('l1_points', l1_points.shape)
        #print('x', x.shape)

        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        se = self.se(x)
        x = x * se
        x = self.drop1(x)
        x = torch.cat([self.skip_connect0(l1_points),x], dim=1)
        x = self.conv2(x)
        return x


class PCT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PCT, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(in_channels=128, out_channels=128, d_tran=128)
        self.sa2 = SA_Layer(in_channels=128, out_channels=128, d_tran=128)
        self.sa3 = SA_Layer(in_channels=128, out_channels=128, d_tran=128)
        self.sa4 = SA_Layer(in_channels=128, out_channels=128, d_tran=128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(128*4, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(3072, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, out_channel, 1)
        # self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        # x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1024 + 64
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1024 + 64
        x = torch.cat((x, x_global_feature), 1)  # 1024 * 3 + 64
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        # x = self.convs3(x)
        # x = F.log_softmax(x, dim=1)

        return x


class SA_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, d_tran=128, headnum=1):
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
        self.headnum = headnum

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        # savepath = 'D:\YSC\experiments\Ysc\Pointnet_Pointnet2_pytorch\dis.pth'
        # state = {
        #     'xyz': xyz,
        #     'point': attention,
        # }
        # torch.save(state, savepath)
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x
