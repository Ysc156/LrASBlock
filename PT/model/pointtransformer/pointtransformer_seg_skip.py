import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from .PCT_skip import PCT
import random

              
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)                 
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1_ = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2_ = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1_ = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2_ = nn.Sequential(nn.Linear(2*in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            #print('x_before', x.shape)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2_(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1_(x)
        else:
            p1, x1, o1 = pxo1
            #print('x_before_1', x1.shape)
            p2, x2, o2 = pxo2
            #print('x_before_2', x2.shape)
            #x = torch.cat([self.linear1(x1), pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)],-1)
            x = self.linear1_(x1) + pointops.interpolation(p2, p1, self.linear2_(x2), o2, o1)
            #print('x_concat',x.shape)
            #print('x_plus',x.shape)
        #print('x_transup', x.shape)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls_ = nn.Sequential(nn.Linear(planes[0]*2, planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], k))
        self.PCT_skip = nn.ModuleList([PCT(planes[i], planes[i]) for i in range(0, len(planes) - 1, 1)])
        self.S_Pt = torch.nn.Parameter(torch.randn(256, 32))
        #self.is_head = nn.Sequential(nn.Linear(planes[4], planes[4]*2), nn.BatchNorm1d(planes[4]*2), nn.ReLU(inplace=True))
        
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        
        #Global_pxo = sample_global_points([p1, x1, o1], 128)
        Global_pxo = collect_global_points([p1, x1, o1], self.S_Pt)
        
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        #print('up_x5',x5.shape)
        x5 = torch.cat([x5,x5], dim=-1)
        #print('up_x5_after',x5.shape)
        x4_up = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        #print('up_x4_up',x4_up.shape)
        x4_skip = sparse_global_attention([p4, x4, o4], self.PCT_skip[3], Global_pxo)
        #print('up_x4_skip',x4_skip.shape)
        x4 = torch.cat([x4_up,x4_skip], dim=-1)
        #print('up_x4',x4.shape)
        x3_up = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        #print('up_x3_up',x3_up.shape)
        x3_skip = sparse_global_attention([p3, x3, o3], self.PCT_skip[2], Global_pxo)
        #print('up_x3_skip',x3_skip.shape)
        x3 = torch.cat([x3_up,x3_skip], dim=-1)
        #print('up_x3',x3.shape)
        x2_up = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        #print('up_x2_up',x2_up.shape)
        x2_skip = sparse_global_attention([p2, x2, o2], self.PCT_skip[1], Global_pxo)
        #print('up_x2_skip',x2_skip.shape)
        x2 = torch.cat([x2_up,x2_skip], dim=-1)
        #print('up_x2',x2.shape)
        
        # print('up_x2',x2)
        x1_up = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        #print('up_x1_up',x1_up.shape)
        x1_skip = sparse_global_attention([p1, x1, o1], self.PCT_skip[0], Global_pxo)
        #print('up_x1_skip',x1_skip.shape)
        x1_out = torch.cat([x1_up,x1_skip], dim=-1)
        #print('up_x1',x1_out.shape)
        
        #print('x1_out',x1_out.shape)
        x = self.cls_(x1_out)
        # print('x_out',x)
        # print('x_skip_out', x_skip)
        # print(self.S_Pt)
        return x


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 2, 2, 6, 3], **kwargs)
    return model

def sample_global_points(pxo, n_S):
    p, x, o = pxo  # (n, 3), (n, c), (b)
    n_o, count = [n_S], n_S
    for i in range(1, o.shape[0]):
        count += n_S
        n_o.append(count)
    n_o = torch.cuda.IntTensor(n_o)
    idx = pointops.furthestsampling(p, o, n_o)  # (m)
    
    n_p = p[idx.long(), :]  # (m, 3)
    n_x = x[idx.long(), :]  # (m, c)
    return [n_p, n_x, n_o]


def sparse_global_attention(pxo, PCT_skip, Global_pxo):
    feat_ = []
    # print(x0.shape)
    # print(o0.shape)

    p, x, o = pxo
    Gp, G_x, G_o = Global_pxo
    # print(G_x.shape)
    # print(G_o.shape)

    assert len(o) == len(G_o)
    for j in range(len(o)):
        if j == 0:
            feat_j = x[:o[j]]
            G_x_j = G_x[:G_o[j]]
            p_j = p[:o[j]]
            Gp_j = Gp[:G_o[j]]
        else:
            feat_j = x[o[j - 1]:o[j]]
            G_x_j = G_x[G_o[j - 1]:G_o[j]]
            p_j = p[o[j - 1]:o[j]]
            Gp_j = Gp[G_o[j - 1]:G_o[j]]
            
        # print(feat_j.shape)
        # print(G_x_j.shape)
        # print('featj', feat_j.unsqueeze(0).transpose(1, 2))
        # print('xj', G_x_j.unsqueeze(0).transpose(1, 2))
        feat_.append(
            PCT_skip(feat_j.unsqueeze(0).transpose(1, 2), G_x_j.unsqueeze(0).transpose(1, 2), p_j.unsqueeze(0).transpose(1, 2), Gp_j.unsqueeze(0).transpose(1, 2)))
    x_0 = torch.cat(feat_, dim=1).squeeze(0)
    return x_0
    
    
def collect_global_points(pxo, self_points):
    p, x, o = pxo  # (n, 3), (n, c), b

    o = torch.cat([o[0:1]*0, o], dim=0)
    n_S = self_points.shape[0]
    n_p, n_x, n_o, count = None, None, None, o[0:1].clone()
    count[0] = n_S

    for i in range(0, o.shape[0]-1):
        
        po = p[o[i]:o[i+1], :]  # (n_o, 3)
        xo = x[o[i]:o[i+1], :]  # (n_o, c)

        W_h = torch.matmul(self_points, xo.transpose(0,1))  # (S, n)
        np = torch.matmul(F.softmax(W_h, dim=-1), po)  # (S, 3)
        gamma = 16
        W_g = torch.exp(-gamma*(square_distance_norm(np,po)))  # (S, n)
        W = W_g * W_h  # (S, n)
        nx = torch.matmul(F.softmax(W, dim=-1), xo)  # (S, c)
        
        if n_p == None:
            n_p = np.clone()
            n_x = nx.clone()
            n_o = count.clone()
        else:
            temp = n_p
            n_p = torch.cat([temp, np], dim=0)
            temp = n_x
            n_x = torch.cat([temp, nx], dim=0)
            temp = n_o
            n_o = torch.cat([temp, count], dim=0)
        count[0] += n_S

    return [n_p, n_x, n_o]
    
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
