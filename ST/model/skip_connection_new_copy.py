import torch
from torch import nn
import torch.nn.functional as F
import inspect

class LrASBlockv1(nn.Module):
    def __init__(self, channel_list, in_channel):
        super(LrASBlockv1, self).__init__()
        trans_dim = 128
        num_layer = len(channel_list)
        # self.conv1 = nn.Conv1d(in_channel, trans_dim, 1, bias=False)
        # self.conv2 = nn.ModuleList([nn.Conv1d(channel, trans_dim, 1, bias=False) for channel in channel_list])
        self.in_channel = in_channel
        self.gn1 = nn.GroupNorm(8, in_channel)
        # #self.gn1 = nn.BatchNorm1d(trans_dim)
        self.gn2 = nn.ModuleList([nn.GroupNorm(16, i) for i in channel_list])
        #self.gn2 = nn.BatchNorm1d(trans_dim)
        self.sa = nn.ModuleList([SA_Layer(q_channels=in_channel, kv_channels=channel_list[i], out_channels=in_channel, embed_size=trans_dim) for i in range(num_layer)])

        # self.conv_fuse = nn.Sequential(nn.Conv1d(in_channel*len(self.sa), in_channel, 1, bias=False),
        #                                nn.GroupNorm(4, in_channel),
        #                                nn.LeakyReLU(negative_slope=0.2))

        #self.convs1 = nn.Conv1d(trans_dim*2, 512, 1)
        # self.dp1 = nn.Dropout(0.5)
        # self.convs2 = nn.Conv1d(128, out_channel, 1)
        # self.convs3 = nn.Conv1d(out_channel, self.part_num, 1)
        # self.gns1 = nn.GroupNorm(4, 128)
        #self.gns1 = nn.BatchNorm1d(512)
        # self.gns2 = nn.GroupNorm(4, out_channel)
        #self.gns2 = nn.BatchNorm1d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x, x_global, xyz, xyz_global):

        x_in = x
        # x = x.unsqueeze(0).transpose(1, 2)
        # xyz = xyz.unsqueeze(0).transpose(1, 2)

        x_global = [x for x in x_global if isinstance(x, torch.Tensor)]
        xyz_global = [x for x in xyz_global if isinstance(x, torch.Tensor)]
        
        assert len(x_global) == len(self.sa)
        for i in range(len(self.sa)):
            x_global[i] = x_global[i].unsqueeze(0)
            xyz_global[i] = xyz_global[i].unsqueeze(0)
            x_global[i] = self.relu(self.gn2[i](x_global[i]))  # B, D, N
        x = self.relu(self.gn1(x))
        # print('x_global0', x_global)
        
        # print('x_global', x_global)
       
        x_out = []
        
        for i in range(len(self.sa)):
            # print('loacl', i, x.shape)
            # print(self.sa[i])
            temp_x, weight = self.sa[i](x, x_global[i], xyz, xyz_global[i])
            x_out.append(temp_x)
        x_out = torch.mean(torch.stack(x_out), dim=0)
        # x_out = torch.cat(x_out, dim=1)
        # x_out = (self.conv_fuse(x_out)).permute(0, 2, 1)
        return x_out + x_in, weight
    
        #return (x).permute(0, 2, 1)

class LrASBlockv2(nn.Module):
    def __init__(self, channel_list, in_channel):
        super(LrASBlockv2, self).__init__()
        trans_dim = 128
        num_layer = len(channel_list)
        # self.conv1 = nn.Conv1d(in_channel, trans_dim, 1, bias=False)
        # self.conv2 = nn.ModuleList([nn.Conv1d(channel, trans_dim, 1, bias=False) for channel in channel_list])
        self.in_channel = in_channel
        self.gn1 = nn.GroupNorm(8, in_channel)
        # #self.gn1 = nn.BatchNorm1d(trans_dim)
        self.gn2 = nn.ModuleList([nn.GroupNorm(16, i) for i in channel_list])
        #self.gn2 = nn.BatchNorm1d(trans_dim)
        self.sa = nn.ModuleList([SA_Layer(q_channels=in_channel, kv_channels=channel_list[i], out_channels=in_channel, embed_size=trans_dim) for i in range(num_layer)])

        # self.conv_fuse = nn.Sequential(nn.Conv1d(in_channel*len(self.sa), in_channel, 1, bias=False),
        #                                nn.GroupNorm(4, in_channel),
        #                                nn.LeakyReLU(negative_slope=0.2))

        #self.convs1 = nn.Conv1d(trans_dim*2, 512, 1)
        # self.dp1 = nn.Dropout(0.5)
        # self.convs2 = nn.Conv1d(128, out_channel, 1)
        # self.convs3 = nn.Conv1d(out_channel, self.part_num, 1)
        # self.gns1 = nn.GroupNorm(4, 128)
        #self.gns1 = nn.BatchNorm1d(512)
        # self.gns2 = nn.GroupNorm(4, out_channel)
        #self.gns2 = nn.BatchNorm1d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x, x_global, xyz, xyz_global):

        x_in = x
        # x = x.unsqueeze(0).transpose(1, 2)
        # xyz = xyz.unsqueeze(0).transpose(1, 2)

        x_global = [x for x in x_global if isinstance(x, torch.Tensor)]
        xyz_global = [x for x in xyz_global if isinstance(x, torch.Tensor)]
        
        assert len(x_global) == len(self.sa)
        for i in range(len(self.sa)):
            x_global[i] = x_global[i].unsqueeze(0)
            xyz_global[i] = xyz_global[i].unsqueeze(0)
            x_global[i] = self.relu(self.gn2[i](x_global[i]))  # B, D, N
        x = self.relu(self.gn1(x))
        # print('x_global0', x_global)
        
        # print('x_global', x_global)
       
        x_out = []
        
        for i in range(len(self.sa)):
            # print('loacl', i, x.shape)
            # print(self.sa[i])
            temp_x, weight = self.sa[i](x, x_global[i], xyz, xyz_global[i])
            x_out.append(temp_x)
        x_out = torch.mean(torch.stack(x_out), dim=0)
        # x_out = torch.cat(x_out, dim=1)
        # x_out = (self.conv_fuse(x_out)).permute(0, 2, 1)
        return x_out + x_in, weight


class SA_Layer(nn.Module):
    def __init__(self, q_channels, kv_channels, out_channels, embed_size=16, num_heads=8):
        super(SA_Layer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.projection_dim = embed_size // num_heads
        self.query_projection = nn.Conv1d(q_channels, embed_size, 1)
        self.key_projection = nn.Conv1d(kv_channels, embed_size, 1)
        self.value_projection = nn.Conv1d(kv_channels, embed_size, 1)
        self.q_gn = nn.GroupNorm(8, embed_size)
        self.combine_heads = nn.Linear(embed_size, out_channels)
        
    def attention(self, query, key, value, xyz_q, xyz_kv):
        assert xyz_q.shape[0] == xyz_kv.shape[0] == 1
        score = torch.matmul(query/(self.projection_dim ** 2), key.transpose(-2, -1))
        scaled_score = score
        scaled_score = torch.clamp(scaled_score,max=65503)
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights
        
    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim) # [B, N, H, C//H]
        #print(x.shape)
        return x.transpose(1, 2) # [B, H, N, C//H]
    
    def forward(self, x_q, x_kv, xyz_q, xyz_kv):
        x_in = x_q
        query = self.q_gn(self.query_projection(x_q)).transpose(-1, -2) # [B, N, C]
        key = self.q_gn(self.key_projection(x_kv)).transpose(-1, -2) # [B, N_g, C]
        value = self.q_gn(self.value_projection(x_kv)).transpose(-1, -2) # [B, N_g, C]
        check(query)
        check(key)
        check(value)
        batch_size = 1
        # Split heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        # Attention
        attention, weights = self.attention(query, key, value, xyz_q, xyz_kv)
        check(attention)
        # Combine heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)

        output = self.combine_heads(attention)
        check(output)
        return output.transpose(1, 2) + x_in, weights
        

def square_distance_norm(src, dst):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points , [N, C]
        dst: target points , [M, C]
    Returns:
        dist: per-point square distance, [N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    # -2xy
    dist = -2 * torch.matmul(src, dst.permute(1, 0))

    # x^2 , y^2
    temp = torch.sum(src ** 2, dim=-1)
    dist += torch.sum(src ** 2, dim=-1)[:, None]
    dist += torch.sum(dst ** 2, dim=-1)[None, :]
    # dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    
def check(a):
    if torch.isinf(a).any():
        print("have inf")
        a[torch.isinf(a)] = 0
    if torch.isnan(a).any():
        print("have nan")
        a[torch.isnan(a)] = 0