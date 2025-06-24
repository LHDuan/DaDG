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
enable_amp = True
find_unused_parameters = True
sync_bn = True
seed = 1204
# trainer
train = dict(
    type="MultiDatasetTrainerEval",
)
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
        DaBN_conditions=("ScanNet", "3D_Front", "ScanNet_3D_Front"),
        DaBN_update="zero",
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,
    conditions=("ScanNet", "3D_Front"),
    template="[x]",
    clip_model="ViT-B/16",
    class_name=[
        # Indoor
        "wall", "floor", "cabinet", "bed", "chair", 
        "sofa", "table", "door", "window", "bookshelf", 
        "bookcase", "picture", "counter", "desk", "shelves", 
        "curtain", "dresser", "pillow", "mirror", "ceiling", 
        "refrigerator", "television", "shower curtain", "nightstand", "toilet", 
        "sink", "lamp", "bathtub", "garbagebin", "board", 
        "beam", "column", "clutter", "otherstructure", "otherfurniture", "otherprop",
        # Front
        'Children Cabinet', 'Nightstand', 'Bookcase or jewelry Armoire', 'Wardrobe', 'Tea Table',
        'Corner Table or Side Table', 'Sideboard or Side Cabinet', 'Wine Cabinet', 'TV Stand', 'Drawer Chest or Corner cabinet',
        'Shelf', 'Round End Table', 'Double Bed', 'Bunk Bed', 'Bed Frame',
        'Single bed', 'Kids Bed', 'Dining Chair', 'Lounge Chair or Book-chair or Computer Chair', 'Dressing Chair',
        'Classic Chinese Chair', 'Barstool', 'Dressing Table', 'Dining Table', 'Desk',
        'Three-Seat Sofa or Multi-seat Sofa', 'armchair', 'Two-seat Sofa', 'L-shaped Sofa', 'Lazy Sofa',
        'Chaise Longue Sofa', 'Footstool or Sofastool or Bed End Stool or Stool', 'Pendant Lamp', 'Ceiling Lamp', 'Back',
        'Flue', 'Customized Fixed Furniture', 'Wall Inner', 'Customized Ceiling', 'Cabinet',
        'Light Band', 'Smart Customized Ceiling', 'Floor', 'Customized Platform', 'Customized Furniture',
        'Customized wainscot', 'Window', 'Customized Personalized Model', 'Column', 'clipMesh',
        'Wall Outer', 'Front', 'Hole', 'Sewer Pipe', 'Bay Window',
        'Slab Side', 'Pocket', 'Slab Bottom', 'Beam', 'Cornice',
        'Baseboard', 'Slab Top', 'Wall Top', 'Customized Background Model', 'Door',
        'Wall Bottom', 'Cabinet or Light Band', 'Ceiling', 'Customized Feature Wall', 'Extrusion Customized Ceiling Model',
        'Extrusion Customized Background Wall'
    ],
    source_index=(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27],
        [i for i in range(36, 36 + 71)]
    ),
    common_index=(
        [0, 1, 4, 5, 6, 7, 8, 10],
        [0, 1, 4, 5, 6, 7, 8, 10, 19, 31, 30]
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

ft_pretransform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.5, dropout_application_ratio=0.5),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="NormalizeColor"),
]
ft_pretransform_mix = ft_pretransform 

sc_pretransform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.5, dropout_application_ratio=0.5),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="NormalizeColor"),
]
sc_pretransform_mix = sc_pretransform
# dataset settings
ignore_index = -1

sc_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

source_map_sc = {
    -1: ignore_index,
    0: 0, # "wall",
    1: 1, # "floor",
    2: 2, # "cabinet",
    3: 3, # "bed",
    4: 4, # "chair",
    5: 5, # "sofa",
    6: 6, # "table",
    7: 7, # "door",
    8: 8, # "window",
    9: 9, # "bookshelf",
    10: 10, # "picture",
    11: 11, # "counter",
    12: 12, # "desk",
    13: 13, # "curtain",
    14: 14, # "refridgerator",
    15: 15, # "shower curtain",
    16: 16, # "toilet",
    17: 17, # "sink",
    18: 18, # "bathtub",
    19: ignore_index, # "otherfurniture",
}

common_map_sc = {
    -1: ignore_index,
    0: 0, # "wall",
    1: 1, # "floor",
    2: ignore_index, # "cabinet",
    3: ignore_index, # "bed",
    4: 2, # "chair",
    5: 3, # "sofa",
    6: 4, # "table",
    7: 5, # "door",
    8: 6, # "window",
    9: 7, # "bookshelf",
    10: ignore_index, # "picture",
    11: ignore_index, # "counter",
    12: ignore_index, # "desk",
    13: ignore_index, # "curtain",
    14: ignore_index, # "refridgerator",
    15: ignore_index, # "shower curtain",
    16: ignore_index, # "toilet",
    17: ignore_index, # "sink",
    18: ignore_index, # "bathtub",
    19: ignore_index, # "otherfurniture",
}

sc_common_names = [
    "wall", "floor", "chair", "sofa", "table", 
    "door", "window", "bookshelf"
]

ft_names = [
    'Children Cabinet', 'Nightstand', 'Bookcase or jewelry Armoire', 'Wardrobe', 'Tea Table',
    'Corner Table or Side Table', 'Sideboard or Side Cabinet', 'Wine Cabinet', 'TV Stand', 'Drawer Chest or Corner cabinet',
    'Shelf', 'Round End Table', 'Double Bed', 'Bunk Bed', 'Bed Frame',
    'Single bed', 'Kids Bed', 'Dining Chair', 'Lounge Chair or Book-chair or Computer Chair', 'Dressing Chair',
    'Classic Chinese Chair', 'Barstool', 'Dressing Table', 'Dining Table', 'Desk',
    'Three-Seat Sofa or Multi-seat Sofa', 'armchair', 'Two-seat Sofa', 'L-shaped Sofa', 'Lazy Sofa',
    'Chaise Longue Sofa', 'Footstool or Sofastool or Bed End Stool or Stool', 'Pendant Lamp', 'Ceiling Lamp', 'Back',
    'Flue', 'Customized Fixed Furniture', 'Wall Inner', 'Customized Ceiling', 'Cabinet',
    'Light Band', 'Smart Customized Ceiling', 'Floor', 'Customized Platform', 'Customized Furniture',
    'Customized wainscot', 'Window', 'Customized Personalized Model', 'Column', 'clipMesh',
    'Wall Outer', 'Front', 'Hole', 'Sewer Pipe', 'Bay Window',
    'Slab Side', 'Pocket', 'Slab Bottom', 'Beam', 'Cornice',
    'Baseboard', 'Slab Top', 'Wall Top', 'Customized Background Model', 'Door',
    'Wall Bottom', 'Cabinet or Light Band', 'Ceiling', 'Customized Feature Wall', 'Extrusion Customized Ceiling Model',
    'Extrusion Customized Background Wall'
]

source_map_ft = {
    0: 0, # Children Cabinet
    1: 1, # Nightstand
    2: 2, # Bookcase or jewelry Armoire
    3: 3, # Wardrobe
    4: 4, # Tea Table
    5: 5, # Corner Table or Side Table
    6: 6, # Sideboard or Side Cabinet
    7: 7, # Wine Cabinet
    8: 8, # TV Stand
    9: 9, # Drawer Chest or Corner cabinet
    10: 10, # Shelf
    11: 11, # Round End Table
    12: 12, # Double Bed
    13: 13, # Bunk Bed
    14: 14, # Bed Frame
    15: 15, # Single bed
    16: 16, # Kids Bed
    17: 17, # Dining Chair
    18: 18, # Lounge Chair or Book-chair or Computer Chair
    19: 19, # Dressing Chair
    20: 20, # Classic Chinese Chair
    21: 21, # Barstool
    22: 22, # Dressing Table
    23: 23, # Dining Table
    24: 24, # Desk
    25: 25, # Three-Seat Sofa or Multi-seat Sofa
    26: 26, # armchair
    27: 27, # Two-seat Sofa
    28: 28, # L-shaped Sofa
    29: 29, # Lazy Sofa
    30: 30, # Chaise Longue Sofa
    31: 31, # Footstool or Sofastool or Bed End Stool or Stool
    32: 32, # Pendant Lamp
    33: 33, # Ceiling Lamp
    34: 34, # Back
    35: 35, # Flue
    36: 36, # Customized Fixed Furniture
    37: 37, # Wall Inner
    38: 38, # Customized Ceiling
    39: 39, # Cabinet
    40: 40, # Light Band
    41: 41, # Smart Customized Ceiling
    42: 42, # Floor
    43: 43, # Customized Platform
    44: 44, # Customized Furniture
    45: 45, # Customized wainscot
    46: 46, # Window
    47: 47, # Customized Personalized Model
    48: 48, # Column
    49: 49, # clipMesh
    50: 50, # Wall Outer
    51: 51, # Front
    52: 52, # Hole
    53: 53, # Sewer Pipe
    54: 54, # Bay Window
    55: 55, # Slab Side
    56: 56, # Pocket
    57: 57, # Slab Bottom
    58: 58, # Beam
    59: 59, # Cornice
    60: 60, # Baseboard
    61: 61, # Slab Top
    62: 62, # Wall Top
    63: 63, # Customized Background Model
    64: 64, # Door
    65: 65, # Wall Bottom
    66: 66, # Cabinet or Light Band
    67: 67, # Ceiling
    68: 68, # Customized Feature Wall
    69: 69, # Extrusion Customized Ceiling Model
    70: 70, # Extrusion Customized Background Wall
}

common_map_ft = {
    0: ignore_index, # Children Cabinet
    1: ignore_index, # Nightstand
    2: 7, # Bookcase or jewelry Armoire
    3: ignore_index, # Wardrobe
    4: 4, # Tea Table
    5: ignore_index, # Corner Table or Side Table
    6: ignore_index, # Sideboard or Side Cabinet
    7: ignore_index, # Wine Cabinet
    8: ignore_index, # TV Stand
    9: ignore_index, # Drawer Chest or Corner cabinet
    10: ignore_index, # Shelf
    11: 4, # Round End Table
    12: ignore_index, # Double Bed
    13: ignore_index, # Bunk Bed
    14: ignore_index, # Bed Frame
    15: ignore_index, # Single bed
    16: ignore_index, # Kids Bed
    17: 2, # Dining Chair
    18: 2, # Lounge Chair or Book-chair or Computer Chair
    19: 2, # Dressing Chair
    20: 2, # Classic Chinese Chair
    21: 2, # Barstool
    22: 4, # Dressing Table
    23: 4, # Dining Table
    24: 4, # Desk
    25: 3, # Three-Seat Sofa or Multi-seat Sofa
    26: 3, # armchair
    27: 3, # Two-seat Sofa
    28: 3, # L-shaped Sofa
    29: 3, # Lazy Sofa
    30: 3, # Chaise Longue Sofa
    31: ignore_index, # Footstool or Sofastool or Bed End Stool or Stool
    32: ignore_index, # Pendant Lamp
    33: ignore_index, # Ceiling Lamp
    34: ignore_index, # Back
    35: ignore_index, # Flue
    36: ignore_index, # Customized Fixed Furniture
    37: 0, # Wall Inner
    38: 8, # Customized Ceiling
    39: ignore_index, # Cabinet
    40: ignore_index, # Light Band
    41: 8, # Smart Customized Ceiling
    42: 1, # Floor
    43: ignore_index, # Customized Platform
    44: ignore_index, # Customized Furniture
    45: ignore_index, # Customized wainscot
    46: 6, # Window
    47: ignore_index, # Customized Personalized Model
    48: 9, # Column
    49: ignore_index, # clipMesh
    50: 0, # Wall Outer
    51: ignore_index, # Front
    52: ignore_index, # Hole
    53: ignore_index, # Sewer Pipe
    54: 6, # Bay Window
    55: ignore_index, # Slab Side
    56: 5, # Pocket
    57: ignore_index, # Slab Bottom
    58: 10, # Beam
    59: ignore_index, # Cornice
    60: 0, # Baseboard
    61: ignore_index, # Slab Top
    62: 0, # Wall Top
    63: 0, # Customized Background Model
    64: 5, # Door
    65: 0, # Wall Bottom
    66: ignore_index, # Cabinet or Light Band
    67: 8, # Ceiling
    68: 0, # Customized Feature Wall
    69: 8, # Extrusion Customized Ceiling Model
    70: 0, # Extrusion Customized Background Wall
}

ft_common_names = [
    "wall", "floor", "chair", "sofa", "table", 
    "door", "window", "bookcase", "ceiling", "column", 
    "beam", 
]

data = dict(
    num_classes=[8, 11],
    ignore_index=-1,
    names=[sc_common_names, ft_common_names],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="MergeDataset",
                split=["train", "train"],
                data_root=["data/scannet", "data/3d_front/train"],
                data_name=["ScanNet", "3D_Front"],
                pre_transform=[
                    sc_pretransform, ft_pretransform
                ],
                ignore_index=-1,
                dataset_idx=1,
                condition_idx=1,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Indoor","Indoor"],
                loop=2,
                ratio=1,
                skip=[1,1],
                mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[source_map_sc, source_map_ft],
                common_mapping=[common_map_sc, common_map_ft],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment", "condition", "dataset_idx", "instance"),
                    ),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "dataset_idx", "instance"),
                        feat_keys=("color",),
                    ),
                ],
            ),
            dict(
                type="MergeDataset",
                split=["train", "train"],
                data_root=["data/scannet", "data/3d_front/train"],
                data_name=["ScanNet", "3D_Front"],
                pre_transform=[
                    sc_pretransform, ft_pretransform
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=0,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Indoor","Indoor"],
                loop=10,
                ratio=1,
                skip=[1,1],
                mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[source_map_sc, source_map_ft],
                common_mapping=[common_map_sc, common_map_ft],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment", "condition", "dataset_idx", "instance"),
                    ),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "dataset_idx", "instance"),
                        feat_keys=("color",),
                    ),
                ],
            ),
            dict(
                type="MergeDataset",
                split=["train", "train"],
                data_root=["data/scannet", "data/3d_front/train"],
                data_name=["ScanNet", "3D_Front"],
                pre_transform=[
                    sc_pretransform, ft_pretransform
                ],
                ignore_index=-1,
                dataset_idx=1,
                condition_idx=2,
                sub_batch=True,
                mix_dataset_idx=[0,1],
                mix_dataset_prob=1.0,
                data_type=["Indoor","Indoor"],
                loop=2,
                ratio=1,
                skip=[1,1],
                mix_prob=0.5,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[source_map_sc, source_map_ft],
                common_mapping=[common_map_sc, common_map_ft],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment", "condition", "dataset_idx", "instance"),
                    ),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "dataset_idx", "instance"),
                        feat_keys=("color",),
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
                data_root=["data/scannet", "data/3d_front/val"],
                data_name=["ScanNet", "3D_Front"],
                pre_transform=[
                    [dict(type="CenterShift", apply_z=True),], 
                    [dict(type="CenterShift", apply_z=True),],
                ],
                ignore_index=-1,
                dataset_idx=0,
                condition_idx=0,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Indoor", "Indoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_sc, common_map_ft],
                common_mapping=[common_map_sc, common_map_ft],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment", "condition", "dataset_idx"),
                    ),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "dataset_idx"),
                        feat_keys=("color",),
                    ),
                ],
            ),
            dict(
                type="MergeDataset",
                split=["val", "val"],
                data_root=["data/scannet", "data/3d_front/val"],
                data_name=["ScanNet", "3D_Front"],
                pre_transform=[
                    [dict(type="CenterShift", apply_z=True),], 
                    [dict(type="CenterShift", apply_z=True),],
                ],
                ignore_index=-1,
                dataset_idx=1,
                condition_idx=1,
                mix_dataset_idx=[0,],
                mix_dataset_prob=0.0,
                data_type=["Indoor", "Indoor"],
                loop=1,
                ratio=1,
                skip=[1,1],
                mix_prob=0.0,
                mix_batch=False,
                mix_same_domain=False,
                mix_cross_domain=False,
                source_mapping=[common_map_sc, common_map_ft],
                common_mapping=[common_map_sc, common_map_ft],
                transform=[
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment", "condition", "dataset_idx"),
                    ),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "dataset_idx"),
                        feat_keys=("color",),
                    ),
                ],
            ),
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
