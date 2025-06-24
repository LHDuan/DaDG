"""
Distribution-aware Normalization
Author: Lunhao Duan (lhduan@whu.edu.cn)
Please cite our work if the code is helpful to you.
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.builder import MODULES


@MODULES.register_module()
class DaNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        norm_update,
        conditions=("ScanNet", "Structured3D", "SemanticKITTI", "nuScenes"),
    ):
        super().__init__()
        self.conditions = conditions
        self.norm_update = norm_update
        self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])

    def was_distance(self, cur_mu, cur_sig, proto_mu, proto_sig):       
        distance = (cur_mu - proto_mu).pow(2) + (cur_sig.pow(2) + proto_sig.pow(2) - 2 * cur_sig * proto_sig)
        return distance

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        condition_list = torch.unique(point.condition)
        
        # Single source dataset
        if len(condition_list) == 1 and condition_list[0] < len(self.conditions):
            point.feat = self.norm[condition_list[0]](point.feat)
        # Multi source dataset or target dataset
        else:
            cur_mu, cur_sig = point.feat.mean(dim=0).detach(), (point.feat.var(dim=0) + self.norm[0].eps).sqrt().detach()
            mean_list = []
            sig_list = []
            for i in range(len(self.conditions)):
                mean_list.append(self.norm[i].running_mean.detach())
                sig_list.append((self.norm[i].running_var + self.norm[i].eps).sqrt().detach())
            proto_mu = torch.stack(mean_list, dim=0)
            proto_sig = torch.stack(sig_list, dim=0)
            distance = self.was_distance(cur_mu, cur_sig, proto_mu, proto_sig)
            distance = distance.mean(dim=1)
            alpha = 1.0 / (1.0 + distance)
            tem_tau = 1.0
            if "soft" in self.norm_update:
                norm_update = self.norm_update.split("_")
                if len(norm_update) > 2:
                    tem_tau = float(norm_update[2])
            alpha = F.softmax(alpha / tem_tau, dim=0)
            # alpha = F.softmax(-distance, dim=0)
            alpha_value, alpha_max = alpha.max(0)
            norm = self.norm[alpha_max]
            if not self.training:
                point.dan_weight += alpha
                point.dan_index[alpha_max] += 1
            if self.training:
                if "soft" in self.norm_update:
                    feat = 0
                    for i in range(len(self.norm)):
                        norm = self.norm[i]
                        if "one" in self.norm_update:
                            momentum = norm.momentum
                        elif "zero"in self.norm_update:
                            momentum = 0.0
                        elif "alpha"in self.norm_update:
                            momentum = norm.momentum * alpha[i]
                        else:
                            raise NotImplementedError
                        feat = feat + alpha[i] * F.batch_norm(point.feat, norm.running_mean, norm.running_var, norm.weight, norm.bias, True, momentum, norm.eps)
                    point.feat = feat                    
                else:
                    if "one" in self.norm_update:
                        momentum = norm.momentum
                    elif "zero"in self.norm_update:
                        momentum = 0.0
                    elif "alpha"in self.norm_update:
                        momentum = norm.momentum * alpha[i]
                    else:
                        raise NotImplementedError
                    # if self.training and momentum > 0:
                    point.feat = F.batch_norm(point.feat, norm.running_mean, norm.running_var, norm.weight, norm.bias, True, momentum, norm.eps)
            else:
                if "soft" in self.norm_update:
                    feat = 0
                    for i in range(len(self.norm)):
                        norm = self.norm[i]
                        feat = feat + alpha[i] * F.batch_norm(point.feat, norm.running_mean, norm.running_var, norm.weight, norm.bias, False, norm.momentum, norm.eps)
                    point.feat = feat
                else:
                    point.feat = F.batch_norm(point.feat, norm.running_mean, norm.running_var, norm.weight, norm.bias, False, norm.momentum, norm.eps)
        return point