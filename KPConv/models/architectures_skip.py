#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
import numpy as np
from .PCT_skip import PCT
import torch.nn.functional as F


def p2p_fitting_regularizer(net):
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        self.PCT_skip = nn.ModuleList()
        self.S_Pt = torch.nn.Parameter(torch.randn(64, 5))

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
                self.PCT_skip.append(PCT(self.encoder_skip_dims[layer], self.encoder_skip_dims[layer]))

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        self_points = self.S_Pt
        Global_xpl = collect_global_points(batch, self_points)
        skip_x = []
        i = 0
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                # print('org_x', x.shape)
                x_temp = sparse_global_attention(x, batch.lengths, self.PCT_skip[3-i], Global_xpl)
                skip_x.append(x_temp)
                i += 1
                # print('skip_', i, x_temp.shape)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


def sparse_global_attention(x0, o0_list, PCT_skip, Global_xpl):
    feat_ = []
    # print(x0.shape)
    # print(o0.shape)
    G_x, _, G_o = Global_xpl
    for i in range(len(o0_list)):
        if o0_list[i].sum() == x0.shape[0]:
            o0 = o0_list[i].clone().detach()
            break
    # print(G_x.shape)
    # print(G_o.shape)

    assert len(o0) == len(G_o)
    for j in range(len(o0)):
        if j == 0:
            feat_j = x0[:o0[j]]
            G_x_j = G_x[:G_o[j]]
        else:
            feat_j = x0[o0[:j].sum():o0[:j+1].sum()]
            G_x_j = G_x[G_o[:j].sum():G_o[:j+1].sum()]
        # print('feat', feat_j.shape)
        # print('x', G_x_j.shape)
        # print(feat_j.unsqueeze(0).transpose(1, 2).shape)
        # print(G_x_j.unsqueeze(0).transpose(1, 2).shape)
        feat_.append(
            PCT_skip(feat_j.unsqueeze(0).transpose(1, 2), G_x_j.unsqueeze(0).transpose(1, 2)))
    x_0 = torch.cat(feat_, dim=1).squeeze(0)
    return x_0


def collect_global_points(batch, self_points):
    feats = batch.features.clone().detach()   # (n, c)
    points = batch.points[0].clone().detach()  # (n, 3)
    lengths = batch.lengths[0].clone().detach()  # (b)
    lengths_O = lengths.clone()
    lengths = torch.cat([lengths[0:1] * 0, lengths], dim=0)
    n_S = self_points.shape[0]
    feats_O, points_O = None, None

    for i in range(1, lengths.shape[0]):

        p = points[lengths[:i].sum():lengths[:i+1].sum(), :]  # (n_o, 3)
        x = feats[lengths[:i].sum():lengths[:i+1].sum(), :]  # (n_o, c)
        # np

        W_h = torch.matmul(self_points, x.transpose(0, 1))  # (S, n)
        W_h = F.softmax(W_h, dim=-1)  # (S, n)
        np = torch.matmul(W_h, p)  # (S, 3)
        # nx
        gamma = 16
        W_g = torch.exp(-gamma * (square_distance_norm(np, p)))  # (S, n)
        W_g = F.softmax(W_g, dim=-1)
        W = W_g * W_h  # (S, n)
        nx = torch.matmul(W, x)  # (S, c)

        # print('W_G', W_g)
        # print('W_h', W_h)
        if feats_O == None:
            points_O = np.clone()
            feats_O = nx.clone()
            lengths_O[i-1] = n_S
        else:
            temp = points_O
            points_O = torch.cat([temp, np], dim=0)
            temp = feats_O
            feats_O = torch.cat([temp, nx], dim=0)
            lengths_O[i-1] = n_S

    return [feats_O, points_O, lengths_O]


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
    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    
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