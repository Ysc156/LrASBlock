import torch
from torch import nn
import torch.nn.functional as F
import inspect

class PCT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PCT, self).__init__()
        self.part_num = 50
        trans_dim = 128
        self.conv1 = nn.Conv1d(in_channel, trans_dim, 1, bias=False)
        # self.conv2 = nn.Conv1d(3, trans_dim, 1, bias=False)
        
        self.gn1 = nn.GroupNorm(16, trans_dim)
        #self.gn1 = nn.BatchNorm1d(trans_dim)
        # self.gn2 = nn.GroupNorm(16, trans_dim)
        #self.gn2 = nn.BatchNorm1d(trans_dim)

        self.sa1 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, embed_size=trans_dim)
        self.sa2 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, embed_size=trans_dim)
        # self.sa3 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, embed_size=trans_dim)
        # self.sa4 = SA_Layer(in_channels=trans_dim, out_channels=trans_dim, embed_size=trans_dim)



        self.conv_fuse = nn.Sequential(nn.Conv1d(trans_dim*2, out_channel, 1, bias=False),
                                       nn.GroupNorm(4, out_channel),
                                       nn.LeakyReLU(negative_slope=0.2))

        #self.convs1 = nn.Conv1d(trans_dim*2, 512, 1)
        # self.dp1 = nn.Dropout(0.5)
        # self.convs2 = nn.Conv1d(128, out_channel, 1)
        # self.convs3 = nn.Conv1d(out_channel, self.part_num, 1)
        # self.gns1 = nn.GroupNorm(4, 128)
        #self.gns1 = nn.BatchNorm1d(512)
        # self.gns2 = nn.GroupNorm(4, out_channel)
        #self.gns2 = nn.BatchNorm1d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x_global, x, xyz_global, xyz):
        # check(x)
        # check(x_global)
        x_in = x
        # batch_size, _, N = x.size()
        x = self.relu(self.gn1(self.conv1(x)))  # B, D, N
        # print('x_global0', x_global)
        x_global = self.relu(self.gn1(self.conv1(x_global)))
        # print('x_global', x_global)
        
        x1 = self.sa1(x, x_global, xyz, xyz_global)
        #check(x1)
        x2 = self.sa2(x1, x_global, xyz, xyz_global)
        #check(x2)
        # x3 = self.sa3(x2, x_global, xyz, xyz_global)
        # check(x3)
        # x4 = self.sa4(x3, x_global, xyz, xyz_global)
        # check(x4)
        x = x1        
        x = torch.cat((x1, x2), dim=1)
        x = (self.conv_fuse(x))
        return (x + x_in).permute(0, 2, 1)
        #return (x).permute(0, 2, 1)
    
class SA_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, embed_size=16, num_heads=8):
        super(SA_Layer, self).__init__()
        
        
        #self.act = nn.ReLU()
        #self.softmax = nn.Softmax(dim=-1)
        #self.pos_xyz = nn.Conv1d(3, d_tran, kernel_size=1, bias=False)
        #self.pos_xyz_G = nn.Conv1d(3, d_tran, kernel_size=1, bias=False)
        
        #assert d_tran % headnum == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.projection_dim = embed_size // num_heads
        self.query_projection = nn.Conv1d(in_channels, embed_size, 1)
        # self.key_projection = nn.Conv1d(in_channels, embed_size, 1)
        # self.value_projection = nn.Conv1d(in_channels, embed_size, 1)
        self.q_gn = nn.GroupNorm(8, embed_size)
        # self.k_gn = nn.GroupNorm(8, embed_size)
        # self.query_projection.weight = self.key_projection.weight
        # self.v_gn = nn.GroupNorm(8, embed_size)
        #self.after_norm = nn.GroupNorm(4, out_channels)
        #self.after_norm = nn.BatchNorm1d(out_channels)
        self.combine_heads = nn.Linear(embed_size, out_channels)
        
    def attention(self, query, key, value, xyz_q, xyz_kv):
        #print('q', query.shape)
        #print('k', key.shape)
        #print('v', value.shape)
        assert xyz_q.shape[0] == xyz_kv.shape[0] == 1
        dist_w_norm = torch.norm((torch.max(xyz_q.squeeze(0),0)[0] - torch.min(xyz_q.squeeze(0),0)[0]), 2)/20
        
        dist_w = square_distance_norm(xyz_q.squeeze(0).transpose(0,1),xyz_kv.squeeze(0).transpose(0,1))//dist_w_norm
        
        w_d = 0.5**dist_w
        
        score = torch.matmul(query/(self.projection_dim ** 2), key.transpose(-2, -1))
        scaled_score = score
        scaled_score = torch.clamp(scaled_score,max=65503)
        #check(scaled_score)
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights
        
    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim) # [B, N, H, C//H]
        #print(x.shape)
        return x.transpose(1, 2) # [B, H, N, C//H]
    
    def forward(self, x_q, x_kv, xyz_q, xyz_kv):
        x_in = x_q
        #batch_size, point_num, _ = inputs.size()
        # Linear projections
        query = self.q_gn(self.query_projection(x_q)).transpose(-1, -2) # [B, N, C]
        key = self.q_gn(self.query_projection(x_kv)).transpose(-1, -2) # [B, N_g, C]
        value = self.q_gn(self.query_projection(x_kv)).transpose(-1, -2) # [B, N_g, C]
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
        #print('q', query)
        #print('k', key)
        #print('v', value)
        #print('atten', attention)
        # Combine heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)

        output = self.combine_heads(attention)
        check(output)
        return output.transpose(1, 2) + x_in
        
    #def forward(self, x_q, x_kv, xyz_q, xyz_kv):
    #    x_in = x_q
        #xyz_q = self.pos_xyz(xyz_q).permute(0, 2, 1)
        #xyz_kv = self.pos_xyz_G(xyz_kv)
    #    x_q = self.q_bn(self.q_conv(x_q)).permute(0, 2, 1)  # b, n1, c
    #    x_k = self.q_bn(self.k_conv(x_kv)) # b, c, n2
    #    x_v = self.v_bn(self.v_conv(x_kv)) # b, c, n2
    #    d_trans = x_q.shape[-1]
    #    q = torch.stack(torch.split(x_q, d_trans//self.num_heads, dim=2), dim=0) # [H, B, N, C//H]
    #    k = torch.stack(torch.split(x_k, d_trans//self.num_heads, dim=1), dim=0) # [H, B, C//H, N_g]
    #    v = torch.stack(torch.split(x_v, d_trans//self.num_heads, dim=1), dim=0) # [H, B, C//H, N_g]
    #    
    #    energy = q*(q.shape[-1]**-0.5) @ k  # [H, B, N, N_g]
    #    
    #    assert xyz_q.shape[0] == xyz_kv.shape[0] == 1
    #    dist_w_norm = torch.norm((torch.max(xyz_q.squeeze(0),0)[0] - torch.min(xyz_q.squeeze(0),0)[0]), 2)/20
        #print(xyz_q.squeeze(0).transpose(0,1).shape)
        #print(xyz_kv.squeeze(0).transpose(0,1).shape)
    #    dist_w = square_distance_norm(xyz_q.squeeze(0).transpose(0,1),xyz_kv.squeeze(0).transpose(0,1))//dist_w_norm
        
    #    w_d = 0.5**dist_w
        
    #    attention = self.softmax(energy)*w_d.unsqueeze(0).unsqueeze(0)  # b, n1, n2
        
    #    v = v.permute(0, 1, 3, 2)  # [H, B, N_g, C//H]
    #    x_r = attention @ v  # [H, B, N, C//H]
    #    print('1', x_r.shape)
    #    x_r = torch.cat(torch.split(x_r, 1, dim=0), dim=3).squeeze(0) #[B,N,C]
    #    x_r = self.act(self.after_norm(self.trans_conv(x_r)))
    #    print(x_r.shape)
        
        
        
    #    x = x_in + x_r.transpose(1,2)
    #    return x

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
    if torch.isinf(a).any() or torch.isnan(a).any():
        torch.set_printoptions(profile="full") 
        aaaaa