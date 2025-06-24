"""
Point Cloud Domain Generalization
Author: Lunhao Duan (lhduan@whu.edu.cn)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
import json
import random
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
import pickle
from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from typing import List
debug_prob = 0.0

class PolarMix(object):
    def __init__(self,
                 swap_ratio: float = 1.0,
                 rotate_paste_ratio: float = 1.0,
                 prob: float = 1.0) -> None:
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio
        self.prob = prob

    def __call__(self, data_dict: dict, data_dict_another: dict, instance_classes: List[int]) -> dict:
        mix_dict = {}
        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -np.arctan2(data_dict["coord"][:, 1], data_dict["coord"][:, 0])
            yaw_another = -np.arctan2(data_dict_another["coord"][:, 1], data_dict_another["coord"][:, 0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            idx_another = (yaw_another > start_angle) & (yaw_another < end_angle)

            # swap
            for key in data_dict.keys():
                mix_dict[key] = np.concatenate((data_dict[key][idx], data_dict_another[key][idx_another]))
        else:
            for key in data_dict.keys():
                mix_dict[key] = data_dict[key]

        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_dict = {}
            for key in data_dict.keys():
                instance_dict[key] = []
            
            for instance_class in instance_classes:
                instance_idx = data_dict_another["segment"] == instance_class
                for key in data_dict.keys():
                    instance_dict[key].append(data_dict_another[key][instance_idx])
            
            for key in data_dict.keys():
                instance_dict[key] = np.concatenate(instance_dict[key], axis=0)

            # rotate-copy
            copy_dict = {}
            for key in data_dict.keys():
                copy_dict[key] = []
            angle_list = [
                np.random.random() * np.pi * 2 / 3,
                (np.random.random() + 1) * np.pi * 2 / 3
            ]
            for angle in angle_list:
                instance_points = instance_dict["coord"].copy()
                rot_cos, rot_sin = np.cos(angle), np.sin(angle)
                rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
                instance_points = np.dot(instance_points, np.transpose(rot_t))
                
                copy_dict["coord"].append(instance_points)
                for key in data_dict.keys():
                    if key != "coord":
                        copy_dict[key].append(instance_dict[key].copy())

            for key in data_dict.keys():
                copy_dict[key] = np.concatenate(copy_dict[key], axis=0)
            
            for key in data_dict.keys():
                mix_dict[key] = np.concatenate((mix_dict[key], copy_dict[key]))
        
        return mix_dict


class RhoMix(object):
    def __init__(self,
                 num_areas=[4, 5, 6],
                 prob=1.0,):
        self.num_areas = num_areas
        self.prob = prob

    def __call__(self, input_dict: dict, data_dict_another: dict) -> dict:
        rho = np.sqrt(input_dict["coord"][:, 0]**2 + input_dict["coord"][:, 1]**2)
        rho_another = np.sqrt(data_dict_another["coord"][:, 0]**2 + data_dict_another["coord"][:, 1]**2)
        
        rho_max = max(np.max(rho), np.max(rho))
        rho_min = min(np.min(rho_another), np.min(rho_another))
        
        num_areas = np.random.choice(self.num_areas, size=1)[0]
        rho_list = np.linspace(rho_min, rho_max, num_areas + 1)
        mix_dict = {}
        for key in input_dict.keys():
            mix_dict[key] = []
        for i in range(num_areas):
            # convert angle to radian
            start_rho = rho_list[i]
            end_rho = rho_list[i+1]
            if i % 2 == 0:  # pick from original point cloud
                idx = (rho > start_rho) & (rho <= end_rho)
                for key in input_dict.keys():
                    mix_dict[key].append(input_dict[key][idx])
            else:  # pickle from mixed point cloud
                idx = (rho_another > start_rho) & (rho_another <= end_rho)
                for key in input_dict.keys():
                    mix_dict[key].append(data_dict_another[key][idx])

        for key in mix_dict.keys():
            mix_dict[key] = np.concatenate(mix_dict[key], axis=0)
        return mix_dict


@DATASETS.register_module()
class MergeDataset(Dataset):
    def __init__(
        self,
        split=["train"],
        data_root=["data/scannet"],
        data_name=("ScanNet"),
        pre_transform=None,
        transform=None,
        ignore_index=-1,
        dataset_idx=0,
        mix_dataset_idx=[0,],
        mix_dataset_prob=0.8,
        data_type=["Indoor",],
        loop=1,
        ratio=1,
        skip=[1,1,1,1],
        mix_prob=0.8,
        mix_batch=False,
        mix_same_domain=False,
        mix_cross_domain=False,
        mix_cross_dataset=True,
        source_mapping=None,
        common_mapping=None,
        instance_list=None,
        rho_mix_prob=0.0,
        polar_mix_prob=0.5,
        sub_batch=False,
        condition_idx=2,
    ):
        super(MergeDataset, self).__init__()
        self.sweeps = 10
        self.loop = loop
        self.ratio = ratio
        self.mix_prob = mix_prob
        self.mix_batch = mix_batch
        self.mix_same_domain = mix_same_domain
        self.mix_cross_domain = mix_cross_domain
        self.mix_cross_dataset = mix_cross_dataset
        self.data_name = data_name
        self.data_type = data_type
        self.dataset_idx = dataset_idx
        self.mix_dataset_idx = mix_dataset_idx
        self.mix_dataset_prob = mix_dataset_prob
        self.data_root = data_root
        self.split = split
        self.skip = skip
        self.rho_mix_prob = rho_mix_prob
        self.polar_mix_prob = polar_mix_prob
        self.sub_batch = sub_batch
        self.condition_idx = condition_idx

        self.pre_transform = [Compose(t) for t in pre_transform]
        self.transform = Compose(transform)

        self.data_list = []
        for name in self.data_name:
            if "ScanNet" in name:
                self.data_list.append(self.get_data_list_sc(name))
            if "3D_Front" in name:
                self.data_list.append(self.get_data_list_ft(name))
            if "S3DIS" in name:
                self.data_list.append(self.get_data_list_s3(name))
            if "SemanticKITTI" in name:
                self.data_list.append(self.get_data_list_sk(name))
            if "nuScenes" in name:
                self.data_list.append(self.get_data_list_nu(name))
            if "SynLiDAR" in name:
                self.data_list.append(self.get_data_list_synlidar(name))
            if "Waymo" in name:
                self.data_list.append(self.get_data_list_wa(name))

        self.ignore_index = ignore_index
        self.source_mapping = source_mapping
        self.common_mapping = common_mapping
        self.instance_list = instance_list
        self.polar_mix = PolarMix()
        self.rho_mix = RhoMix()

        logger = get_root_logger()
        logger.info(
            "Totally {}: {} samples * {} loop.".format(
                self.data_name[self.dataset_idx], len(self.data_list[self.dataset_idx]), self.loop
            )
        )

    def load_mapper_file(self, map_dict=None):
        if map_dict is not None:
            class_names = map_dict['classes']
            src_classes = map_dict['src']
            remapper = np.ones(256, dtype=np.int32) * (self.ignore_index)
            for l0 in src_classes:
                remapper[int(l0)] = class_names.index(src_classes[l0])
            return remapper
        else:
            return None

    def get_data_list_ft(self, name):
        ft_idx = self.data_name.index(name)
        data_list = glob.glob(os.path.join(self.data_root[ft_idx], "*.npy"))
        data_list.sort()
        if self.skip[ft_idx] > 1:
            data_list = data_list[::self.skip[ft_idx]]
        return data_list        

    def get_data_list_sc(self,name):
        sc_idx = self.data_name.index(name)
        if isinstance(self.split[sc_idx], str):
            data_list = glob.glob(os.path.join(self.data_root[sc_idx], self.split[sc_idx], "*", "coord.npy"))
        elif isinstance(self.split[sc_idx], Sequence):
            data_list = []
            for split in self.split[sc_idx]:
                data_list += glob.glob(os.path.join(self.data_root[sc_idx], split[sc_idx], "*", "coord.npy"))
        else:
            raise NotImplementedError
        data_list.sort()
        if self.skip[sc_idx] > 1:
            data_list = data_list[::self.skip[sc_idx]]
        return data_list

    def get_data_list_s3(self,name):
        s3_idx = self.data_name.index(name)
        if isinstance(self.split[s3_idx], str):
            data_list = glob.glob(os.path.join(self.data_root[s3_idx], self.split[s3_idx], "*", "coord.npy"))
        elif isinstance(self.split[s3_idx], Sequence):
            data_list = []
            for split in self.split[s3_idx]:
                data_list += glob.glob(os.path.join(self.data_root[s3_idx], split, "*", "coord.npy"))
        else:
            raise NotImplementedError
        data_list.sort()
        if self.skip[s3_idx] > 1:
            data_list = data_list[::self.skip[s3_idx]]
        return data_list

    def get_data_list_sk(self, name):
        sk_idx = self.data_name.index(name)
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split[sk_idx], str):
            seq_list = split2seq[self.split[sk_idx]]
        elif isinstance(self.split[sk_idx], list):
            seq_list = []
            for split in self.split[sk_idx]:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError
        data_list = []
        txt_file = open(os.path.join(self.data_root[sk_idx], self.split[sk_idx]+".txt"), "r").readlines()
        for line in txt_file:
            data_list.append(line.strip().replace("data/semantic_kitti", self.data_root[sk_idx]))
        # for seq in seq_list:
        #     seq = str(seq).zfill(2)
        #     seq_folder = os.path.join(self.data_root[sk_idx], "dataset", "sequences", seq)
        #     seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        #     data_list += [
        #         os.path.join(seq_folder, "velodyne", file) for file in seq_files
        #     ]
        # data_list.sort()
        if self.skip[sk_idx] > 1:
            data_list = data_list[::self.skip[sk_idx]]
        return data_list

    def get_info_path(self, split, idx):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root[idx], "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root[idx], "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root[idx], "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list_nu(self, name):
        nu_idx = self.data_name.index(name)
        if isinstance(self.split[nu_idx], str):
            info_paths = [self.get_info_path(self.split[nu_idx], nu_idx)]
        elif isinstance(self.split[nu_idx], Sequence):
            info_paths = [self.get_info_path(s, nu_idx) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        txt_file = open(os.path.join(self.data_root[nu_idx], self.split[nu_idx]+".txt"), "r").readlines()
        for line in txt_file:
            data_list.append(
                {
                    "lidar_path": line.strip().split(",")[0],
                    "gt_segment_path": line.strip().split(",")[1],
                }
            )
        # for info_path in info_paths:
        #     with open(info_path, "rb") as f:
        #         info = pickle.load(f)
        #         info = sorted(info, key = lambda e:e['lidar_path'], reverse = False)
        #         data_list.extend(info)
        if self.skip[nu_idx] > 1:
            data_list = data_list[::self.skip[nu_idx]]
        return data_list

    def get_data_list_synlidar(self, name):
        synlidar_idx = self.data_name.index(name)
        split2seq = dict(
            train=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            val=[2],
        )
        if isinstance(self.split[synlidar_idx], str):
            seq_list = split2seq[self.split[synlidar_idx]]
        elif isinstance(self.split[synlidar_idx], list):
            seq_list = []
            for split in self.split[synlidar_idx]:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError
        data_list = []
        txt_file = open(os.path.join(self.data_root[synlidar_idx], self.split[synlidar_idx]+".txt"), "r").readlines()
        for line in txt_file:
            data_list.append(line.strip().replace("data/synlidar", self.data_root[synlidar_idx]))
        # for seq in seq_list:
        #     seq = str(seq).zfill(2)
        #     seq_folder = os.path.join(self.data_root[synlidar_idx], seq)
        #     seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        #     data_list += [
        #         os.path.join(seq_folder, "velodyne", file) for file in seq_files
        #     ]
        # data_list.sort()
        if self.skip[synlidar_idx] > 1:
            data_list = data_list[::self.skip[synlidar_idx]]
        return data_list

    def get_data_list_wa(self, name):
        wa_idx = self.data_name.index(name)
        data_list = []
        txt_file = open(os.path.join(self.data_root[wa_idx], self.split[wa_idx]+".txt"), "r").readlines()
        for line in txt_file:
            data_list.append(line.strip().replace("data/waymo", self.data_root[wa_idx]))
        # if isinstance(self.split[wa_idx], str):
        #     data_list = glob.glob(os.path.join(self.data_root[wa_idx], self.split[wa_idx], "*", "*", "coord.npy"))
        # elif isinstance(self.split[wa_idx], Sequence):
        #     data_list = []
        #     for split in self.split[wa_idx]:
        #         data_list += glob.glob(os.path.join(self.data_root[wa_idx], split, "*", "*", "coord.npy"))
        # else:
        #     raise NotImplementedError
        # data_list.sort()

        if self.skip[wa_idx] > 1:
            data_list = data_list[::self.skip[wa_idx]]
        return data_list

    def get_data_ft(self, idx, name):
        ft_idx = self.data_name.index(name)
        data_path = self.data_list[ft_idx][idx % len(self.data_list[ft_idx])]
        data = np.load(data_path)
        coord = np.ascontiguousarray(data[:, :3]).astype(np.float32)
        color = np.ascontiguousarray(data[:, 3:6]).astype(np.float32)
        normal = np.zeros((coord.shape[0], 3)).astype(np.float32)
        strength = np.zeros((coord.shape[0], 3)).astype(np.float32)
        segment = np.ascontiguousarray(data[:, 6], dtype=np.int32)
        if self.source_mapping[ft_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[ft_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[ft_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[ft_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_sc(self, idx, name):
        sc_idx = self.data_name.index(name)
        data_path = self.data_list[sc_idx][idx % len(self.data_list[sc_idx])]
        if ".npy" not in data_path:
            data = torch.load(data_path)
            coord = data["coord"].astype(np.float32)
            color = data["color"].astype(np.float32)
            normal = data["normal"].astype(np.float32)
            strength = np.zeros((coord.shape[0], 3)).astype(np.float32)
            segment = data["semantic_gt20"].reshape([-1]).astype(np.int32)
        else:
            coord_path = self.data_list[sc_idx][idx % len(self.data_list[sc_idx])]
            coord = np.load(coord_path).astype(np.float32)
            color = np.load(coord_path.replace("coord.npy", "color.npy")).astype(np.float32)
            normal = np.load(coord_path.replace("coord.npy", "normal.npy")).astype(np.float32)
            strength = np.zeros((coord.shape[0], 3)).astype(np.float32)
            segment = np.load(coord_path.replace("coord.npy", "segment20.npy")).reshape(-1).astype(np.int32)
        if self.source_mapping[sc_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[sc_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[sc_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[sc_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_s3(self, idx, name):
        s3_idx = self.data_name.index(name)
        coord_path = self.data_list[s3_idx][idx % len(self.data_list[s3_idx])]
        coord = np.load(coord_path).astype(np.float32)
        color = np.load(coord_path.replace("coord.npy", "color.npy")).astype(np.float32)
        normal = np.load(coord_path.replace("coord.npy", "normal.npy")).astype(np.float32)
        strength = np.zeros((coord.shape[0], 3)).astype(np.float32)
        segment = np.load(coord_path.replace("coord.npy", "segment.npy")).reshape(-1).astype(np.int32)
        if self.source_mapping[s3_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[s3_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[s3_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[s3_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_sk(self, idx, name):
        sk_idx = self.data_name.index(name)
        data_path = self.data_list[sk_idx][idx % len(self.data_list[sk_idx])]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])
        strength = np.ones_like(strength)
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = (segment & 0xFFFF).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        color = np.zeros((coord.shape[0], 3)).astype(np.float32)
        normal = np.zeros((coord.shape[0], 3)).astype(np.float32)
        if self.source_mapping[sk_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[sk_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[sk_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[sk_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_nu(self, idx, name):
        nu_idx = self.data_name.index(name)
        data = self.data_list[nu_idx][idx % len(self.data_list[nu_idx])]
        lidar_path = os.path.join(self.data_root[nu_idx], "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]
        strength = np.ones_like(strength)
        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root[nu_idx], "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        color = np.zeros((coord.shape[0], 3)).astype(np.float32)
        normal = np.zeros((coord.shape[0], 3)).astype(np.float32)
        if self.source_mapping[nu_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[nu_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[nu_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[nu_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_synlidar(self, idx, name):
        synlidar_idx = self.data_name.index(name)
        data_path = self.data_list[synlidar_idx][idx % len(self.data_list[synlidar_idx])]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])
        strength = np.ones_like(strength)
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        color = np.zeros((coord.shape[0], 3)).astype(np.float32)
        normal = np.zeros((coord.shape[0], 3)).astype(np.float32)
        if self.source_mapping[synlidar_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[synlidar_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[synlidar_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[synlidar_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_wa(self, idx, name):
        wa_idx = self.data_name.index(name)
        coord_path = self.data_list[wa_idx][idx % len(self.data_list[wa_idx])]
        coord = np.load(coord_path).astype(np.float32)
        color = np.zeros((coord.shape[0], 3)).astype(np.float32)
        normal = np.zeros((coord.shape[0], 3)).astype(np.float32)
        strength = np.load(coord_path.replace("coord.npy", "strength.npy")).reshape([-1, 1]).astype(np.float32)
        strength = np.ones_like(strength)
        segment = np.load(coord_path.replace("coord.npy", "segment.npy")).reshape(-1).astype(np.int32)
        if self.source_mapping[wa_idx] is not None:
            segment_source = np.vectorize(self.source_mapping[wa_idx].__getitem__)(segment).astype(np.int32)
        if self.common_mapping[wa_idx] is not None:
            segment_common = np.vectorize(self.common_mapping[wa_idx].__getitem__)(segment).astype(np.int32)
        condition = np.zeros_like(segment_source) + self.condition_idx
        dataset_idx = np.zeros_like(segment_source) + self.data_name.index(name.split("_sub")[0])
        data_dict = dict(coord=coord, normal=normal, color=color, strength=strength, dataset_idx=dataset_idx,
                         segment=segment_source, condition=condition, instance=segment_common)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def get_data(self, idx, dataset_idx):
        name = self.data_name[dataset_idx]
        if "3D_Front" in name:
            data_dict = self.get_data_ft(idx, name)
            data_dict = self.pre_transform[dataset_idx](data_dict)
            data_dict["coord"] = data_dict["coord"] * 2.5

        if "ScanNet" in name:
            data_dict = self.get_data_sc(idx, name)
            data_dict = self.pre_transform[dataset_idx](data_dict)
            data_dict["coord"] = data_dict["coord"] * 2.5

        if "S3DIS" in name:
            data_dict = self.get_data_s3(idx,name)
            data_dict = self.pre_transform[dataset_idx](data_dict)
            data_dict["coord"] = data_dict["coord"] * 2.5
        
        if "SemanticKITTI" in name:
            data_dict = self.get_data_sk(idx, name)
            data_dict = self.pre_transform[dataset_idx](data_dict)
        
        if "nuScenes" in name:
            data_dict = self.get_data_nu(idx, name)
            data_dict = self.pre_transform[dataset_idx](data_dict)

        if "SynLiDAR" in name:
            data_dict = self.get_data_synlidar(idx, name)
            data_dict = self.pre_transform[dataset_idx](data_dict)

        if "Waymo" in name:
            data_dict = self.get_data_wa(idx, name)
            data_dict = self.pre_transform[dataset_idx](data_dict)

        return data_dict

    def mix_two_data_dict(self, data_dict, dataset_idx, data_dict_another, mix_data_idx):
        if self.data_type[dataset_idx] == "Indoor" and self.data_type[mix_data_idx] == "Indoor":
            data_dict = self.mix_indoor(data_dict, dataset_idx, data_dict_another, mix_data_idx)
        if self.data_type[dataset_idx] == "Outdoor" and self.data_type[mix_data_idx] == "Outdoor":
            data_dict = self.mix_outdoor(data_dict, dataset_idx, data_dict_another, mix_data_idx)
        return data_dict

    def mix_dataset(self, data_dict, dataset_idx):
        if random.random() < self.mix_dataset_prob and len(self.mix_dataset_idx) > 0:
            mix_data_idx = random.sample(self.mix_dataset_idx, 1)[0]
            if self.mix_same_domain:
                while mix_data_idx == dataset_idx or self.data_type[dataset_idx] != self.data_type[mix_data_idx]:
                    mix_data_idx = random.sample(self.mix_dataset_idx, 1)[0]
            elif self.mix_cross_domain:
                while mix_data_idx == dataset_idx or self.data_type[dataset_idx] == self.data_type[mix_data_idx]:
                    mix_data_idx = random.sample(self.mix_dataset_idx, 1)[0]
            elif self.mix_cross_dataset:
                while mix_data_idx == dataset_idx:
                    mix_data_idx = random.sample(self.mix_dataset_idx, 1)[0]
            data_dict_another = self.get_data(random.randint(0, len(self.data_list[mix_data_idx]) - 1), mix_data_idx)
            data_dict = self.mix_two_data_dict(data_dict, dataset_idx, data_dict_another, mix_data_idx)
        return data_dict

    def mix_data(self, data_dict, dataset_idx):
        if random.random() < self.mix_prob:
            data_dict_another = self.get_data(random.randint(0, len(self.data_list[dataset_idx]) - 1), dataset_idx)
            data_dict = self.mix_two_data_dict(data_dict, dataset_idx, data_dict_another, dataset_idx)
        return data_dict

    def mix_indoor(self, data_dict, dataset_idx, data_dict_another, mix_data_idx):
        x_min, y_min, z_min = data_dict_another["coord"].min(axis=0)
        x_max, y_max, z_max = data_dict_another["coord"].max(axis=0)
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        data_dict_another["coord"] -= shift

        x_min, y_min, z_min = data_dict["coord"].min(axis=0)
        x_max, y_max, z_max = data_dict["coord"].max(axis=0)
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
        data_dict["coord"] -= shift

        shift = np.random.uniform(data_dict_another["coord"].min(0), data_dict_another["coord"].max(0)) / 2
        shift[2] = 0
        data_dict_another["coord"] += shift
        for key in data_dict.keys():
            data_dict[key] = np.concatenate((data_dict[key], data_dict_another[key]))

        if random.random() < debug_prob:
            print("indoormix_mix3d")
            color = data_dict["color"]
            color[data_dict['condition']==dataset_idx] = [170,203,237]
            color[data_dict['condition']==mix_data_idx] = [255,186,117]
            coord_color = np.concatenate((data_dict["coord"], color), axis=1)
            np.savetxt("./indoormix_mix3d_{}_{}_{}.txt".format(len(data_dict["coord"]), dataset_idx, mix_data_idx), coord_color, fmt='%.1f')
        return data_dict

    def mix_outdoor(self, data_dict, dataset_idx, data_dict_another, mix_data_idx):
        if random.random() < self.rho_mix_prob:
            data_dict = self.rho_mix(data_dict, data_dict_another)
        elif random.random() < self.polar_mix_prob:
            data_dict = self.polar_mix(data_dict, data_dict_another, instance_classes=self.instance_list[mix_data_idx])
        else:
            for key in data_dict.keys():
                data_dict[key] = np.concatenate((data_dict[key], data_dict_another[key]))

        if random.random() < debug_prob:
            print("outdoor_mix3d")
            color = data_dict["color"]
            color[data_dict['condition']==dataset_idx] = [170,203,237]
            color[data_dict['condition']==mix_data_idx] = [255,186,117]
            coord_color = np.concatenate((data_dict["coord"], color), axis=1)
            np.savetxt("./outdoor_mix3d_{}_{}_{}.txt".format(len(data_dict["coord"]), dataset_idx, mix_data_idx), coord_color, fmt='%.1f')
        return data_dict

    def __getitem__(self, idx):
        if self.mix_batch :
            dataset_idx = random.sample(self.mix_dataset_idx, 1)[0]
            idx = random.randint(0, len(self.data_list[dataset_idx]) - 1)
            data_dict = self.get_data(idx, dataset_idx)
            raw_coord_len = len(data_dict["coord"])
            data_dict = self.mix_dataset(data_dict, dataset_idx)
            mix_dataset_len = len(data_dict["coord"])
            data_dict = self.transform(data_dict)
            voxel_len = len(data_dict["coord"])
        elif self.sub_batch:
            dataset_idx = random.sample(self.mix_dataset_idx, 1)[0]
            idx = random.randint(0, len(self.data_list[dataset_idx]) - 1)
            data_dict = self.get_data(idx, dataset_idx)
            data_dict = self.mix_data(data_dict, dataset_idx)
            data_dict = self.transform(data_dict)
        else:
            data_dict = self.get_data(idx, self.dataset_idx)
            raw_coord_len = len(data_dict["coord"])
            data_dict = self.mix_data(data_dict, self.dataset_idx)
            mix_data_len = len(data_dict["coord"])
            data_dict = self.transform(data_dict)
            voxel_len = len(data_dict["coord"])
        return data_dict

    def __len__(self):
        return int(len(self.data_list[self.dataset_idx]) * self.loop)