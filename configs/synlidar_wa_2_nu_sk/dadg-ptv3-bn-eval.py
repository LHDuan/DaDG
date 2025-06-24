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
        DaBN_conditions=("SynLiDAR", "Waymo", "SynLiDAR_Waymo"),
        DaBN_update="zero",
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,
    conditions=("SynLiDAR", "Waymo"),
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
        [i for i in range(18 + 13, 18 + 13 + 21)]
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

wa_pretransform_val = [
    dict(type="RandomShift", shift=((0, 0), (0, 0), (0, 0))),
    dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -2, 75.2, 75.2, 4)),
]
wa_pretransform = [dict(type="RandomBeamDropout", beam_sampling_ratio=[0.4, 0.6], 
    dropout_application_ratio=0.5, beam_num=64),] + wa_pretransform_val + [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
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

wa_names = [
    "car", "truck", "other vehicle", "motorcyclist", "bicyclist", 
    "person", "traffic sign", "traffic light", "traffic pole", "construction cone", 
    "bicycle", "motorcycle", "building", "vegetation", "trunk vegetation", 
    "sidewalk curb", "road", "lane marker road", "other ground road", "terrain",
    "sidewalk",
]
wa_instance=[
    0,1,2,3,4,5,6,7,8,9,10,11
]
source_map_wa = {
    -1: ignore_index,
    0:0, #Car
    1:1, #Truck
    2:2, #Bus Other Vehicle
    3:2, #Other Vehicle
    4:3, #Motorcyclist
    5:4, #Bicyclist
    6:5, #person
    7:6, #Sign
    8:7, #Traffic Light
    9:8, #Pole
    10:9, #Construction Cone
    11:10, #Bicycle
    12:11, #Motorcycle
    13:12, #Building
    14:13, #Vegetation
    15:14, #Tree Trunk
    16:15, #Curb
    17:16, #Road
    18:17, #Lane Marker
    19:18, #Other Ground
    20:19, #Walkable
    21:20, #Sidewalk
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

data = dict(
    num_classes=[10,10],
    ignore_index=-1,
    names=[common_names,common_names],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="MergeDataset",
                split=["train", "training"],
                data_root=["data/synlidar", "data/waymo"],
                data_name=["SynLiDAR", "Waymo"],
                pre_transform=[
                    synlidar_pretransform, 
                    wa_pretransform, 
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
                mix_prob=0.8,
                rho_mix_prob=0.0,
                polar_mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[
                    source_map_synlidar, 
                    source_map_wa
                ],
                common_mapping=[
                    common_map_synlidar, 
                    common_map_wa
                ],
                instance_list=[
                    synlidar_instance,
                    wa_instance
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
                split=["train", "training"],
                data_root=["data/synlidar", "data/waymo"],
                data_name=["SynLiDAR", "Waymo"],
                pre_transform=[
                    synlidar_pretransform, 
                    wa_pretransform, 
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
                mix_prob=0.8,
                rho_mix_prob=0.0,
                polar_mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[
                    source_map_synlidar, 
                    source_map_wa
                ],
                common_mapping=[
                    common_map_synlidar, 
                    common_map_wa
                ],
                instance_list=[
                    synlidar_instance,
                    wa_instance
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
                split=["train", "training"],
                data_root=["data/synlidar", "data/waymo"],
                data_name=["SynLiDAR", "Waymo"],
                pre_transform=[
                    synlidar_pretransform, 
                    wa_pretransform, 
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=2,
                sub_batch=True,
                mix_dataset_idx=[0,1],
                mix_dataset_prob=0.8,
                data_type=["Outdoor","Outdoor","Outdoor","Outdoor"],
                loop=1,
                ratio=1,
                skip=[1,1,1,1],
                mix_prob=0.8, 
                rho_mix_prob=0.0, 
                polar_mix_prob=0.5, 
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                mix_cross_dataset=False,
                source_mapping=[
                    source_map_synlidar, 
                    source_map_wa
                ],
                common_mapping=[
                    common_map_synlidar, 
                    common_map_wa
                ],
                instance_list=[
                    synlidar_instance,
                    wa_instance
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
                split=["val", "validation"],
                data_root=["data/synlidar", "data/waymo"],
                data_name=["SynLiDAR", "Waymo"],
                pre_transform=[
                    synlidar_pretransform_val,
                    wa_pretransform_val,
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
                source_mapping=[common_map_synlidar, common_map_wa],
                common_mapping=[common_map_synlidar, common_map_wa],
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
                split=["val", "validation"],
                data_root=["data/synlidar", "data/waymo"],
                data_name=["SynLiDAR", "Waymo"],
                pre_transform=[
                    synlidar_pretransform_val,
                    wa_pretransform_val,
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
                source_mapping=[common_map_synlidar, common_map_wa],
                common_mapping=[common_map_synlidar, common_map_wa],
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
            )
        ]
    ),
    test=dict(
        type="ScanNetDataset",
        split="val",
        data_root="data/scannet",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "normal"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="Add", keys_dict={"condition": "ScanNet"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)
