"""
DaDG with PTv3-BN
Author: Lunhao Duan (lhduan@whu.edu.cn)
Please cite our work if the code is helpful to you.
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size = 8  # bs: total bs in all gpus
batch_size_val = 8
num_worker = 8

mix_prob = 0.0
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
find_unused_parameters = True
sync_bn = True
seed = 1204
# trainer
train = dict(
    type="MultiDatasetTrainerEval",
)
# Tester
test = dict(type="MultiSemSegEvaluator", verbose=True)
batch_size_test = 4

evaluate = True  # evaluate after each epoch training process
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluatorMultiDataset"),
    dict(type="CheckpointSaver", save_freq=None),
]

# model settings
model = dict(
    type="DaDG-v1m1",
    backbone=dict(
        type="PT-v3-bn",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        DaBN_conditions=("SynLiDAR", "SemanticKITTI", "SynLiDAR_SemanticKITTI"),
        DaBN_update="zero",
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,
    conditions=("SynLiDAR", "SemanticKITTI"),
    template="[x]",
    clip_model="ViT-B/16",
    class_name=[
        # SemanticKITTI 18 class
        "car", "bicycle", "motorcycle", "truck", "other vehicle",
        "pedestrian", "person who rides a bicycle", "person who rides a motorcycle", "road", "parking road",
        "sidewalk", "building", "fence", "vegetation", "trunk vegetation", 
        "terrain", "pole", "traffic sign",
        # nuScenes 13 class
        "barrier", "bicycle", "other vehicle", "car", "motorcycle", 
        "pedestrian", "traffic cone", "truck", "road", "sidewalk", 
        "terrain", "man made", "vegetation",
        # waymo 21 class
        "car", "truck", "other vehicle", "person who rides a motorcycle", "person who rides a bicycle", 
        "pedestrian", "sign", "traffic light", "pole", "construction cone", 
        "bicycle", "motorcycle", "building", "vegetation", "trunk vegetation", 
        "curb sidewalk", "road", "lane marker road", "other ground road", "terrain",
        "sidewalk",
        #synlidar 28 class
        "car", "pick-up truck", "truck", "other vehicle", "bicycle",
        "motorcycle", "road", "sidewalk", "parking road", "female pedestrian", 
        "male pedestrian", "kid pedestrian", "crowd pedestrian", "person who rides a bicycle", "person who rides a motorcycle", 
        "building", "vegetation", "trunk vegetation", "terrain", "traffic sign", 
        "pole", "traffic cone", "fence", #"garbage can", "electric box", 
        # "table", "chair", "bench", 
        # common class
        'car', 'bicycle', 'motorcycle', 'truck', 'other vehicle',
        'pedestrian', 'road', 'sidewalk', 'terrain', 'vegetation',
    ],
    source_index=(
        [i for i in range(18 + 13 + 21, 18 + 13 + 21 + 23)],
        [i for i in range(18)]
    ),
    common_index=(
        [i for i in range(18 + 13 + 21 + 23, 18 + 13 + 21 + 23 + 10)],
        [i for i in range(18 + 13 + 21 + 23, 18 + 13 + 21 + 23 + 10)]
    ),
)
ignore_index=-1
# scheduler settings
epoch = 10
eval_epoch = 10
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005, 0.0005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0005)]

synlidar_pretransform_val = [
    dict(type="RandomShift", shift=((0, 0), (0, 0), (2.0, 2.0))),
    dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -2, 75.2, 75.2, 4)),
]
synlidar_pretransform = [dict(type="RandomBeamDropout", beam_sampling_ratio=[0.4, 0.6], 
    dropout_application_ratio=0.5, beam_num=64),] + synlidar_pretransform_val + [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
]
sk_pretransform_val = [
    dict(type="RandomShift", shift=((0, 0), (0, 0), (1.8, 1.8))),
    dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -2, 75.2, 75.2, 4)),
]
sk_pretransform = [dict(type="RandomBeamDropout", beam_sampling_ratio=[0.4, 0.6], 
    dropout_application_ratio=0.5, beam_num=64),] + sk_pretransform_val + [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
]

nu_pretransform_val = [
    dict(type="RandomShift", shift=((0, 0), (0, 0), (1.8, 1.8))),
    dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -2, 75.2, 75.2, 4)),
]

wa_pretransform_val = [
    dict(type="RandomShift", shift=((0, 0), (0, 0), (0, 0))),
    dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -2, 75.2, 75.2, 4)),
]

# dataset settings
ignore_index = -1
common_instance = [0,1,2,3,4,5]

synlidar_names = [
    "car", "pick-up truck", "truck", "other vehicle", "bicycle",
    "motorcycle", "road", "sidewalk", "parking road", "female pedestrian", 
    "male pedestrian", "kid pedestrian", "crowd pedestrian", "person who rides a bicycle", "person who rides a motorcycle", 
    "building", "vegetation", "trunk vegetation", "terrain", "traffic sign", 
    "pole", "traffic cone", "fence", #"garbage can", "electric box", 
    # "table", "chair", "bench", 
]
synlidar_instance = [
    0,1,2,3,4,5,9,10,11,12,13,14
]
source_map_synlidar = {
    -1: ignore_index,
    0 : ignore_index, #"unlabeled",
    1: 0, #"car",
    2: 1, #"pick-up",
    3: 2, #"truck",
    4: 3, #"bus", -> "other-vehicle"
    5: 4, #"bicycle",
    6: 5, #"motorcycle",
    7: 3, #"other-vehicle",
    8: 6, #"road",
    9: 7, #"sidewalk",
    10: 8, #"parking",
    11: ignore_index, #"other-ground",
    12: 9, #"female",
    13: 10, #"male",
    14: 11, #"kid",
    15: 12, #"crowd",  # multiple person that are very close
    16: 13, #"bicyclist",
    17: 14, #"motorcyclist",
    18: 15, #"building",
    19: ignore_index, #"other-structure",
    20: 16, #"vegetation",
    21: 17, #"trunk",
    22: 18, #"terrain",
    23: 19, #"traffic-sign",
    24: 20, #"pole",
    25: 21, #"traffic-cone",
    26: 22, #"fence",
    27: ignore_index, #"garbage-can",
    28: ignore_index, #"electric-box",
    29: ignore_index, #"table",
    30: ignore_index, #"chair",
    31: ignore_index, #"bench",
    32: ignore_index, #"other-object",
}
common_map_synlidar = {
    -1: ignore_index,
    0 : ignore_index, #"unlabeled",
    1: 0, #"car",
    2: 3, #"pick-up",
    3: 3, #"truck",
    4: 4, #"bus",
    5: 1, #"bicycle",
    6: 2, #"motorcycle",
    7: 4, #"other-vehicle",
    8: 6, #"road",
    9: 7, #"sidewalk",
    10: 6, #"parking",
    11: ignore_index, #"other-ground",
    12: 5, #"female",
    13: 5, #"male",
    14: 5, #"kid",
    15: 5, #"crowd",  # multiple person that are very close
    16: ignore_index, #"bicyclist",
    17: ignore_index, #"motorcyclist",
    18: ignore_index, #"building",
    19: ignore_index, #"other-structure",
    20: 9, #"vegetation",
    21: 9, #"trunk",
    22: 8, #"terrain",
    23: ignore_index, #"traffic-sign",
    24: ignore_index, #"pole",
    25: ignore_index, #"traffic-cone",
    26: ignore_index, #"fence",
    27: ignore_index, #"garbage-can",
    28: ignore_index, #"electric-box",
    29: ignore_index, #"table",
    30: ignore_index, #"chair",
    31: ignore_index, #"bench",
    32: ignore_index, #"other-object",
}

sk_names = [
    "car", "bicycle", "motorcycle", "truck", "other vehicle",
    "pedestrian", "person who rides a bicycle", "person who rides a motorcycle", "road", "parking road",
    "sidewalk", "building", "fence", "vegetation", "trunk vegetation", 
    "terrain", "pole", "traffic sign",
]
sk_instance = [
    0,1,2,3,4,5,6,7
]
source_map_sk = {
    -1: ignore_index,
    0 : ignore_index,     # "unlabeled"
    1 : ignore_index,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,    # "car"
    11: 1,     # "bicycle"
    13: 4,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,     # "motorcycle"
    16: 4,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,     # "truck"
    20: 4,     # "other-vehicle"
    30: 5,     # "person"
    31: 6,     # "bicyclist"
    32: 7,     # "motorcyclist"
    40: 8,    # "road"
    44: 9,    # "parking"
    48: 10,    # "sidewalk"
    49: ignore_index,    # "other-ground"
    50: 11,    # "building"
    51: 12,    # "fence"
    52: ignore_index,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,     # "lane-marking" to "road" ---------------------------------mapped
    70: 13,    # "vegetation"
    71: 14,    # "trunk"
    72: 15,    # "terrain"
    80: 16,    # "pole"
    81: 17,    # "traffic-sign"
    99: -1,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,    # "moving-car" to "car" ------------------------------------mapped
    253: 6,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,    # "moving-person" to "person" ------------------------------mapped
    255: 7,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,    # "moving-truck" to "truck" --------------------------------mapped
    259: 4,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
common_map_sk = {
    -1: ignore_index,
    0 : ignore_index,     # "unlabeled"
    1 : ignore_index,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,    # "car"
    11: 1,     # "bicycle"
    13: 4,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,     # "motorcycle"
    16: 4,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,     # "truck"
    20: 4,     # "other-vehicle"
    30: 5,     # "person"
    31: ignore_index,     # "bicyclist"
    32: ignore_index,     # "motorcyclist"
    40: 6,    # "road"
    44: 6,    # "parking"
    48: 7,    # "sidewalk"
    49: ignore_index,    # "other-ground"
    50: ignore_index,    # "building"
    51: ignore_index,    # "fence"
    52: ignore_index,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 6,     # "lane-marking" to "road" ---------------------------------mapped
    70: 9,    # "vegetation"
    71: 9,    # "trunk"
    72: 8,    # "terrain"
    80: ignore_index,    # "pole"
    81: ignore_index,    # "traffic-sign"
    99: -1,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,    # "moving-car" to "car" ------------------------------------mapped
    253: ignore_index,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,    # "moving-person" to "person" ------------------------------mapped
    255: ignore_index,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,    # "moving-truck" to "truck" --------------------------------mapped
    259: 4,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

common_map_nu = {
    0: ignore_index, #'noise'
    1: ignore_index, #'animal'
    2: 5, #'human.person.adult'
    3: 5, #'human.person.child'
    4: 5, #'human.person.construction_worker'
    5: ignore_index, #'human.person.personal_mobility'
    6: 5, #'human.person.police_officer'
    7: ignore_index, #'human.person.stroller'
    8: ignore_index, #'human.person.wheelchair'
    9: ignore_index, #'movable_object.barrier'
    10: ignore_index, #'movable_object.debris'
    11: ignore_index, #'movable_object.pushable_pullable'
    12: ignore_index, #'movable_object.trafficcone'
    13: ignore_index, #'static_object.bicycle_rack'
    14: 1, #'vehicle.bicycle'
    15: 4, #'vehicle.bus.bendy' mapped to "other-vehicle"
    16: 4, #'vehicle.bus.rigid' mapped to "other-vehicle"
    17: 0, #'vehicle.car'
    18: 4, #'vehicle.construction' mapped to "other-vehicle"
    19: ignore_index, #'vehicle.emergency.ambulance'
    20: ignore_index, #'vehicle.emergency.police'
    21: 2, #'vehicle.motorcycle'
    22: 4, #'vehicle.trailer' mapped to "other-vehicle"
    23: 3, #'vehicle.truck'
    24: 6, #'flat.driveable_surface'
    25: ignore_index, #'flat.other'
    26: 7, #'flat.sidewalk'
    27: 8, #'flat.terrain'
    28: ignore_index, #'static.manmade'
    29: ignore_index, #'static.other'
    30: 9, #'static.vegetation'
    31: ignore_index, #'vehicle.ego'
}

common_map_wa = {
    -1: ignore_index,
    0:0, #Car
    1:3, #Truck
    2:4, #Bus
    3:4, #Other Vehicle
    4:ignore_index, #Motorcyclist
    5:ignore_index, #Bicyclist
    6:5, #person
    7:ignore_index, #Sign
    8:ignore_index, #Traffic Light
    9:ignore_index, #Pole
    10:ignore_index, #Construction Cone
    11:1, #Bicycle
    12:2, #Motorcycle
    13:ignore_index, #Building
    14:9, #Vegetation
    15:9, #Tree Trunk
    16:7, #Curb
    17:6, #Road
    18:6, #Lane Marker
    19:6, #Other Ground
    20:8, #Walkable
    21:7, #Sidewalk
}

common_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other vehicle',
    'person', 'road', 'sidewalk', 'terrain', 'vegetation',
]

test_knn_transform = [
    dict(
        type="Copy",
        keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
    ),
    dict(
        type="GridSample",
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        keys=("coord", "segment", "condition", "strength", "dataset_idx"),
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord", "grid_coord", "origin_coord", "segment", "origin_segment", 
            "condition", "strength", "dataset_idx"
        ),
        offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
        feat_keys=("coord", ),
    ),
]

data = dict(
    num_classes=[10,10,10,10],
    ignore_index=-1,
    names=[nu_names,nu_names,nu_names,nu_names],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="MergeDataset",
                split=["train", "train"],
                data_root=["data/synlidar", "data/semantic_kitti"],
                data_name=["SynLiDAR", "SemanticKITTI"],
                pre_transform=[
                    synlidar_pretransform, 
                    sk_pretransform, 
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=0,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.5,
                rho_mix_prob=0.0,
                polar_mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[
                    source_map_synlidar, 
                    source_map_sk, 
                ],
                common_mapping=[
                    common_map_synlidar, 
                    common_map_sk, 
                ],
                instance_list=[
                    synlidar_instance,
                    sk_instance,
                ],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "segment", "condition", "instance", "strength", "dataset_idx"),
                    ),
                    dict(type="SphereCrop", point_max=153600, mode="random"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "instance", "strength", "dataset_idx"),
                        feat_keys=("coord", ),
                    ),
                ],
            ),
            dict(
                type="MergeDataset",
                split=["train", "train"],
                data_root=["data/synlidar", "data/semantic_kitti"],
                data_name=["SynLiDAR", "SemanticKITTI"],
                pre_transform=[
                    synlidar_pretransform, 
                    sk_pretransform, 
                ],
                ignore_index=-1,
                dataset_idx=1,
                condition_idx=1,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.5,
                rho_mix_prob=0.0,
                polar_mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[
                    source_map_synlidar, 
                    source_map_sk, 
                ],
                common_mapping=[
                    common_map_synlidar, 
                    common_map_sk, 
                ],
                instance_list=[
                    synlidar_instance,
                    sk_instance,
                ],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "segment", "condition", "instance", "strength", "dataset_idx"),
                    ),
                    dict(type="SphereCrop", point_max=153600, mode="random"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "instance", "strength", "dataset_idx"),
                        feat_keys=("coord", ),
                    ),
                ],
            ),
            dict(
                type="MergeDataset",
                split=["train", "train"],
                data_root=["data/synlidar", "data/semantic_kitti"],
                data_name=["SynLiDAR", "SemanticKITTI"],
                pre_transform=[
                    synlidar_pretransform, 
                    sk_pretransform, 
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=2,
                sub_batch=True,
                mix_dataset_idx=[0,1],
                mix_dataset_prob=0.5, 
                data_type=["Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.5, 
                rho_mix_prob=0.0, 
                polar_mix_prob=0.5, 
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                mix_cross_dataset=False,
                source_mapping=[
                    source_map_synlidar, 
                    source_map_sk, 
                ],
                common_mapping=[
                    common_map_synlidar, 
                    common_map_sk, 
                ],
                instance_list=[
                    synlidar_instance,
                    sk_instance,
                ],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "segment", "condition", "instance", "strength", "dataset_idx"),
                    ),
                    dict(type="SphereCrop", point_max=153600, mode="random"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "instance", "strength", "dataset_idx"),
                        feat_keys=("coord", ),
                    ),
                ],
            ),
        ],
    ),
    val=dict(
        type="ConcatDataset",
        datasets = [
            dict(
                type="MergeDataset",
                split=["val", "val"],
                data_root=["data/synlidar", "data/semantic_kitti"],
                data_name=["SynLiDAR", "SemanticKITTI"],
                pre_transform=[
                    synlidar_pretransform_val, 
                    sk_pretransform_val, 
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=0,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_synlidar, common_map_sk],
                common_mapping=[common_map_synlidar, common_map_sk],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "segment", "condition", "strength", "dataset_idx"),
                    ),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "strength", "dataset_idx"),
                        feat_keys=("coord", ),
                    ),
                ],
            ),
            dict(
                type="MergeDataset",
                split=["val", "val"],
                data_root=["data/synlidar", "data/semantic_kitti"],
                data_name=["SynLiDAR", "SemanticKITTI"],
                pre_transform=[
                    synlidar_pretransform_val, 
                    sk_pretransform_val, 
                ],
                ignore_index=-1,
                dataset_idx=1,
                condition_idx=1,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_synlidar, common_map_sk],
                common_mapping=[common_map_synlidar, common_map_sk],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "segment", "condition", "strength", "dataset_idx"),
                    ),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "strength", "dataset_idx"),
                        feat_keys=("coord", ),
                    ),
                ],
            ),
        ]
    ),
    test=dict(
        type="ConcatDataset",
        condition_idx_list=[0],
        datasets = [ 
            dict(
                type="MergeDataset",
                split=["val", "val", "val", "validation"],
                data_root=["data/synlidar", "data/semantic_kitti", "data/nuscenes", "data/waymo"],
                data_name=["SynLiDAR", "SemanticKITTI", "nuScenes", "Waymo"],
                pre_transform=[
                    synlidar_pretransform_val, 
                    sk_pretransform_val, 
                    nu_pretransform_val, 
                    wa_pretransform_val
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=0, 
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor","Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1,1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                common_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                transform=test_knn_transform,
            ),
            dict(
                type="MergeDataset",
                split=["val", "val", "val", "validation"],
                data_root=["data/synlidar", "data/semantic_kitti", "data/nuscenes", "data/waymo"],
                data_name=["SynLiDAR", "SemanticKITTI", "nuScenes", "Waymo"],
                pre_transform=[
                    synlidar_pretransform_val, 
                    sk_pretransform_val, 
                    nu_pretransform_val, 
                    wa_pretransform_val
                ],
                ignore_index=-1,
                dataset_idx=1,
                condition_idx=1, 
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor","Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1,1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                common_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                transform=test_knn_transform,
            ),
            dict(
                type="MergeDataset",
                split=["val", "val", "val", "validation"],
                data_root=["data/synlidar", "data/semantic_kitti", "data/nuscenes", "data/waymo"],
                data_name=["SynLiDAR", "SemanticKITTI", "nuScenes", "Waymo"],
                pre_transform=[
                    synlidar_pretransform_val, 
                    sk_pretransform_val, 
                    nu_pretransform_val, 
                    wa_pretransform_val
                ],
                ignore_index=-1,
                dataset_idx=2,
                condition_idx=3, 
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor","Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1,1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                common_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                transform=test_knn_transform,
            ),
            dict(
                type="MergeDataset",
                split=["val", "val", "val", "validation"],
                data_root=["data/synlidar", "data/semantic_kitti", "data/nuscenes", "data/waymo"],
                data_name=["SynLiDAR", "SemanticKITTI", "nuScenes", "Waymo"],
                pre_transform=[
                    synlidar_pretransform_val, 
                    sk_pretransform_val, 
                    nu_pretransform_val, 
                    wa_pretransform_val
                ],
                ignore_index=-1,
                dataset_idx=3,
                condition_idx=3, 
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Outdoor","Outdoor","Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1,1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                common_mapping=[common_map_synlidar, common_map_sk, 
                                common_map_nu, common_map_wa],
                transform=test_knn_transform,
            ),
        ]
    ),
)
