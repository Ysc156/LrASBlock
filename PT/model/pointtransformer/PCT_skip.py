import torch
from torch import nn
import torch.nn.functional as F


class PCT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PCT, self).__init__()
        self.part_num = 50
        trans_dim = 64
        self.conv1 = nn.Conv1d(in_channel, trans_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, trans_dim, kernel_size=1, bias=False)
        
        self.gn1 = nn.GroupNorm(16, trans_dim)
        self.gn2 = nn.GroupNorm(16, trans_dim)

        self.sa1 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, d_tran=trans_dim)
        self.sa2 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, d_tran=trans_dim)
        self.sa3 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, d_tran=trans_dim)
        self.sa4 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, d_tran=trans_dim)



        self.conv_fuse = nn.Sequential(nn.Conv1d(trans_dim*4, trans_dim*2, kernel_size=1, bias=False),
                                       nn.GroupNorm(16, trans_dim*2),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(trans_dim*2, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, out_channel, 1)
        self.convs3 = nn.Conv1d(out_channel, self.part_num, 1)
        self.gns1 = nn.GroupNorm(8, 512)
        self.gns2 = nn.GroupNorm(4, out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x, x_global, xyz, xyz_global):
        # print("x", x.shape)
        # print("x_global", x_global.shape)
        batch_size, _, N = x.size()
        x = self.relu(self.gn1(self.conv1(x)))  # B, D, N
        # print('x_global0', x_global)
        x_global = self.relu(self.gn2(self.conv2(x_global)))
        # print('x_global', x_global)
        
        x1 = self.sa1(x, x_global, xyz, xyz_global)
        # print('x1', x1)
        x2 = self.sa2(x1, x_global, xyz, xyz_global)
        x3 = self.sa3(x2, x_global, xyz, xyz_global)
        x4 = self.sa4(x3, x_global, xyz, xyz_global)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        #x_max = torch.max(x, 2)[0]
        #x_avg = torch.mean(x, 2)
        #x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        #x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.view(batch_size, trans_dim, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        # x_global_feature = torch.concat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1024 + 64
        #x_global_feature = torch.cat((x_max_feature, x_avg_feature), dim=1)  # 1024 + 64
        #x = torch.cat((x, x_global_feature), dim=1)  # 1024 * 3 + 64
        x = self.relu(self.gns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.gns2(self.convs2(x)))
        return x.permute(0, 2, 1)


class SA_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, d_tran=16, headnum=1):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(in_channels, d_tran, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, d_tran, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(in_channels, d_tran, 1)
        self.trans_conv = nn.Conv1d(d_tran, out_channels, 1)
        self.after_norm = nn.GroupNorm(4, out_channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.headnum = headnum
        self.pos_xyz = nn.Conv1d(3, d_tran, kernel_size=1, bias=False)
        self.pos_xyz_G = nn.Conv1d(3, d_tran, kernel_size=1, bias=False)

    def forward(self, x_q, x_kv, xyz_q, xyz_kv):
        xyz_q = self.pos_xyz(xyz_q).permute(0, 2, 1)
        xyz_kv = self.pos_xyz_G(xyz_kv)
        x_q = self.q_conv(x_q).permute(0, 2, 1)+xyz_q  # b, n1, c
        # print('xkv', x_kv)
        x_k = self.k_conv(x_kv) + xyz_kv # b, c, n2
        x_v = self.v_conv(x_kv) +xyz_kv # b, c, n2
        
        # print('xk', x_k)
        energy = torch.bmm(x_q, x_k)  # b, n1, n2
        # print('energy', energy)
        # print('energy.min', energy.min())
        # print('energy.max', energy.max())
        # print('energy', energy.shape)
        attention = self.softmax(energy)  # b, n1, n2
        # print('attention0', attention)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # b, n1, n2
        # print('attention', attention)
        x_v = x_v.permute(0, 2, 1)  # b, n2, c
        x_r = torch.bmm(attention, x_v)  # b, c, n1
        # print('x_r0', x_r)
        x_r = self.act(self.after_norm(self.trans_conv((x_q - x_r).permute(0, 2, 1))))
        # print('x_r1', x_r)
        x = x_q.permute(0, 2, 1) + x_r
        return x
