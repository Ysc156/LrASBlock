import os
import numpy as np

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import collate_fn
from util.data_util import data_prepare_semantic_kitti as data_prepare
import glob

class semantic_kitti(Dataset):
    def __init__(self, split='train', data_root='data/semantic_kitti', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        ignore_label = 255
        self.learning_map = {
                0: ignore_label,  # "unlabeled"
                1: ignore_label,  # "outlier" mapped to "unlabeled" --------------------------mapped
                10: 0,  # "car"
                11: 1,  # "bicycle"
                13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
                15: 2,  # "motorcycle"
                16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
                18: 3,  # "truck"
                20: 4,  # "other-vehicle"
                30: 5,  # "person"
                31: 6,  # "bicyclist"
                32: 7,  # "motorcyclist"
                40: 8,  # "road"
                44: 9,  # "parking"
                48: 10,  # "sidewalk"
                49: 11,  # "other-ground"
                50: 12,  # "building"
                51: 13,  # "fence"
                52: ignore_label,  # "other-structure" mapped to "unlabeled" ------------------mapped
                60: 8,  # "lane-marking" to "road" ---------------------------------mapped
                70: 14,  # "vegetation"
                71: 15,  # "trunk"
                72: 16,  # "terrain"
                80: 17,  # "pole"
                81: 18,  # "traffic-sign"
                99: ignore_label,  # "other-object" to "unlabeled" ----------------------------mapped
                252: 0,  # "moving-car" to "car" ------------------------------------mapped
                253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
                254: 5,  # "moving-person" to "person" ------------------------------mapped
                255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
                256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
                257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
                258: 3,  # "moving-truck" to "truck" --------------------------------mapped
                259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
            }



        self.split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        )
        seq_list = self.split2seq[split]
        self.data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "sequences", seq)
            seq_files = sorted(
                os.listdir(os.path.join(seq_folder, "velodyne")))
            if split == "val":
                indices = np.linspace(0, len(seq_files) - 1, 1000, dtype=int)
                sampled_seq_files = [seq_files[i] for i in indices]
                self.data_list += [os.path.join(seq_folder, "velodyne", file) for file in sampled_seq_files]
            # else:
            #     self.data_list += [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
        if split == "val":
            pass
        else:
            seq_folder = os.path.join(self.data_root, "sequences", "08")
            seq_files = sorted(
                    os.listdir(os.path.join(seq_folder, "velodyne")))
            indices = np.linspace(0, len(seq_files) - 1, 1000, dtype=int)
            sampled_seq_files = [seq_files[i] for i in indices]
            self.data_list += [os.path.join(seq_folder, "velodyne", file) for file in sampled_seq_files]
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def prepare_data(self, idx):
        # load data
        data_idx = idx % len(self.data_list)
        with open(self.data_list[data_idx], 'rb') as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                label = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            label = np.zeros(coord.shape[0]).astype(np.int32)
        label = np.vectorize(self.learning_map.__getitem__)(label & 0xFFFF).astype(np.int64)
        coord, feat, label = data_prepare(coord, strength, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def get_data_name(self, idx):
        return self.data_list[self.data_list[idx % len(self.data_list)]]

    def __getitem__(self, idx):

        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
