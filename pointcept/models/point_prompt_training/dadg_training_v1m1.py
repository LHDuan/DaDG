"""
Distribution-aware domain generalization
Author: Lunhao Duan (lhduan@whu.edu.cn)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import clip
import torch.nn as nn
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria

@MODELS.register_module("DaDG-v1m1")
class PointPromptTraining(nn.Module):
    """
    PointPromptTraining provides Data-driven Context and enables multi-dataset training with
    Language-driven Categorical Alignment. PDNorm is supported by SpUNet-v1m3 to adapt the
    backbone to a specific dataset with a given dataset condition and context.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        backbone_out_channels=96,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
        template="[x]",
        clip_model="ViT-B/16",
        class_name=(
            "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
            "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
            "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
            "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
            "clutter", "otherstructure", "otherfurniture", "otherprop",
        ),
        source_index=(
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
            (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
        ),
        common_index=(
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
            (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
        ),
    ):
        super().__init__()
        assert len(conditions) == len(source_index)
        assert len(conditions) == len(common_index)
        assert backbone.type in ["PT-v3-bn"]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.conditions = conditions
        self.source_index = source_index
        self.common_index = common_index
        clip_model, _ = clip.load(
            clip_model, device="cpu", download_root="./.cache/clip"
        )
        clip_model.requires_grad_(False)
        class_prompt = [template.replace("[x]", name) for name in class_name]
        class_token = clip.tokenize(class_prompt)
        class_embedding = clip_model.encode_text(class_token)
        class_embedding = class_embedding / class_embedding.norm(
            dim=-1, keepdim=True
        )
        self.register_buffer("class_embedding", class_embedding)
        self.proj_head = nn.Linear(
            backbone_out_channels, clip_model.text_projection.shape[1]
        )
        self.logit_scale = clip_model.logit_scale

    def forward(self, data_dict):
        # condition = data_dict["condition"][0]
        point = self.backbone(data_dict)
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        feat = self.proj_head(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        loss = 0
        logit_scale = self.logit_scale.exp()
        # data_dict["condition"] = data_dict["condition"] % len(self.conditions)
        # condition_list = torch.unique(data_dict["condition"])
        dataset_list = torch.unique(data_dict["dataset_idx"])
        for dataset_idx in dataset_list:
            dataset_mask = data_dict["dataset_idx"] == dataset_idx
            feat_mask = feat[dataset_mask]
            if self.training:
                if "segment" in data_dict.keys():
                    sim_source = (
                        feat_mask
                        @ self.class_embedding[
                            self.source_index[dataset_idx], :
                        ].t()
                    )
                    seg_source_logits = logit_scale * sim_source
                    loss += self.criteria(seg_source_logits, data_dict["segment"][dataset_mask]) / len(dataset_list)
                if "instance" in data_dict.keys():
                    sim_common = (
                        feat_mask
                        @ self.class_embedding[
                            self.common_index[dataset_idx], :
                        ].t()
                    )
                    seg_common_logits = logit_scale * sim_common
                    loss += self.criteria(seg_common_logits, data_dict["instance"][dataset_mask]) / len(dataset_list)
            else:
                sim_common = (
                    feat_mask
                    @ self.class_embedding[
                        self.common_index[dataset_idx], :
                    ].t()
                )
                seg_logits = logit_scale * sim_common
                loss += self.criteria(seg_logits, data_dict["segment"][dataset_mask]) / len(dataset_list)

        # train
        if self.training:
            return dict(loss=loss)
        # eval or test
        elif "segment" in data_dict.keys():
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_common_logits)
