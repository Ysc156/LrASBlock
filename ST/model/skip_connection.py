import torch
from torch import nn
import torch.nn.functional as F


class PCT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PCT, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = GA_Transformer(in_channel=128, trans_dim=128)
        self.sa2 = GA_Transformer(in_channel=128, trans_dim=128)
        # self.sa3 = SA_Layer(in_channels=128, out_channels=128, d_tran=128)
        # self.sa4 = SA_Layer(in_channels=128, out_channels=128, d_tran=128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(128*2, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        # self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(64),
        #                                 nn.LeakyReLU(negative_slope=0.2))

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
        # x3 = self.sa3(x2)
        # x4 = self.sa4(x3)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        # x_global_feature = torch.concat((x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1024 + 64
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1024 + 64
        x = torch.cat((x, x_global_feature), 1)  # 1024 * 3 + 64
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        # x = self.convs3(x)
        # x = F.log_softmax(x, dim=1)

        return x


class GA_Transformer(nn.Module):
    def __init__(self, in_channel, trans_dim=128):
        super(GA_Transformer, self).__init__()
        out_channel = in_channel
        self.conv_q = nn.Conv1d(in_channel, trans_dim, 1, bias=False)
        self.conv_k = nn.Conv1d(in_channel, trans_dim, 1, bias=False)
        self.conv_v = nn.Conv1d(in_channel, trans_dim, 1, bias=False)
        self.xyz_emb = nn.Conv1d(3, trans_dim, 1, bias=False)
        self.xyz_global_emb = nn.Conv1d(3, trans_dim, 1, bias=False)

        self.cross_attention = CrossAttention(trans_dim, 8)
        
        self.conv_fuse = nn.Conv1d(trans_dim, 512, 1, bias=False)
        self.convs2 = nn.Conv1d(512, out_channel, 1)

    def forward(self, x, x_global, xyz, xyz_global):
        x_in = x
        query = self.conv_q(x).permute(2, 0, 1)  # (N, B, D)
        key = self.conv_k(x_global).permute(2, 0, 1)
        value = self.conv_v(x_global).permute(2, 0, 1)
        xyz_f = self.xyz_emb(xyz).permute(2, 0, 1)
        xyz_global_f = self.xyz_global_emb(xyz_global).permute(2, 0, 1)
        x = self.cross_attention(query + xyz_f, key + xyz_global_f, value)
        x = x.permute(1, 2, 0)  # (B, D, N)
        x = self.conv_fuse(x)
        x = self.convs2(x)

        return x.permute(0, 2, 1) + x_in

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value)
        return attn_output