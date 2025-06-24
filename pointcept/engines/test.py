
"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import pointops

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a",# if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                if "class_embedding" not in key:
                    weight[key] = value
            # for key, value in model.state_dict().items():
            #     # print(key, value.shape)
            #     self.logger.info(f"{key} {value.shape}")
            # aa
            load_info = model.load_state_dict(weight, strict=False)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
            self.logger.info("load info: {}".format(load_info))
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegEvaluator(TesterBase):
    def test(self):
        # assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        logger.info("Set condition to {name}".format(name=self.cfg.data.test.condition_idx))
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_name = str(idx)
            input_dict = data_dict
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            pred = pred.cpu().numpy()
            segment = segment.cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes[0], self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            if idx % 100 == 0:
                logger.info(
                    "Test: {} [{}/{}]-{} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Accuracy {acc:.4f} ({m_acc:.4f}) "
                    "mIoU {iou:.4f} ({m_iou:.4f})".format(
                        data_name,
                        idx + 1,
                        len(self.test_loader),
                        segment.size,
                        batch_time=batch_time,
                        acc=acc,
                        m_acc=m_acc,
                        iou=iou,
                        m_iou=m_iou,
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes[0]):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[0][i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


def compute_confusion_matrix(pred, target, num_classes, ignore_index=-1):
    mask = (target != ignore_index)
    conf_matrix = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return conf_matrix


def confusion_matrix_to_latex(conf_matrix, class_names):
    latex_str = "\\begin{tabular}{l|" + "c" * len(class_names) + "}\n"
    latex_str += " & " + " & ".join(class_names) + " \\\\\n"
    latex_str += "\\hline\n"
    for i, row in enumerate(conf_matrix):
        latex_str += f"{class_names[i]} & " + " & ".join(map(str, row)) + " \\\\\n"
    latex_str += "\\end{tabular}"
    return latex_str


@TESTERS.register_module()
class MultiSemSegEvaluator(SemSegEvaluator):
    def build_test_loader(self):
        test_loader = []
        for dataset_i in self.cfg.data.test.datasets:
            test_dataset_i = build_dataset(dataset_i)
            if comm.get_world_size() > 1:
                test_sampler_i = torch.utils.data.distributed.DistributedSampler(test_dataset_i)
            else:
                test_sampler_i = None
            test_loader_i = torch.utils.data.DataLoader(
                test_dataset_i,
                batch_size=self.cfg.batch_size_test_per_gpu,
                shuffle=False,
                num_workers=self.cfg.batch_size_test_per_gpu,
                pin_memory=True,
                sampler=test_sampler_i,
                collate_fn=self.__class__.collate_fn,
            )
            test_loader.append(test_loader_i)
        return test_loader

    def test(self):
        # assert self.test_loader.batch_size == 1
        # for test_loader_i in self.test_loader:
        for test_loader_ind in range(len(self.test_loader)):
            dataset_idx = self.cfg.data.test.datasets[test_loader_ind].dataset_idx
            test_loader_i = self.test_loader[test_loader_ind]
            logger = get_root_logger()
            logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
            logger.info("Set condition to {name}".format(name=self.cfg.data.test.datasets[test_loader_ind].condition_idx))
            logger.info("Set DaBN_update to {name}".format(name=self.cfg.model.backbone.DaBN_update))
            batch_time = AverageMeter()
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            self.model.eval()

            # confusion_matrix = np.zeros((self.cfg.data.num_classes[dataset_idx], self.cfg.data.num_classes[dataset_idx]), dtype=np.int64)

            save_path = os.path.join(self.cfg.save_path, "result")
            make_dirs(save_path)
            
            comm.synchronize()
            record = {}
            # fragment inference
            dan_weight = torch.zeros(
                (len(self.cfg.model.backbone.DaBN_conditions),)
            ).cuda()
            dan_index = torch.zeros(
                (len(self.cfg.model.backbone.DaBN_conditions),)
            ).cuda()
            for idx, data_dict in enumerate(test_loader_i):
                end = time.time()
                data_name = str(idx)
                input_dict = data_dict
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                if len(self.cfg.data.test.condition_idx_list) > 1:
                    output_dict_list = []
                    for condition_idx in self.cfg.data.test.condition_idx_list:
                        input_dict["condition"] = torch.zeros_like(input_dict["condition"]) + condition_idx
                        with torch.no_grad():
                            output_dict = self.model(input_dict)
                            output_dict_list.append(output_dict)
                    output = output_dict_list[0]["seg_logits"]
                    loss = output_dict_list[0]["loss"]
                    # pred = output.max(1)[1]
                    for condition_idx in range(1, len(self.cfg.data.test.condition_idx_list)):
                        loss = loss + output_dict_list[condition_idx]["loss"]
                        output = output + output_dict_list[condition_idx]["seg_logits"]
                    loss = loss / len(self.cfg.data.test.condition_idx_list)
                    pred = output.max(1)[1]
                else:
                    with torch.no_grad():
                        output_dict = self.model(input_dict)
                    output = output_dict["seg_logits"]
                    loss = output_dict["loss"]

                    # output = torch.softmax(output, dim=1)
                    # output[:,0] += torch.sum(output[:,4:], dim=1)
                    
                    pred = output.max(1)[1]
                    # print(torch.sum(pred >= 4), len(pred))
                    # pred[pred == 14] = 2
                    # pred[pred >= 7] = 1
                    # pred[pred >= 4] = 0
                    
                    
                    dan_weight += output_dict["dan_weight"]
                    dan_index += output_dict["dan_index"]
                    # print(output_dict["dan_index"])

                segment = input_dict["segment"]
                if "origin_coord" in input_dict.keys():
                    knn_idx, _ = pointops.knn_query(
                        1,
                        input_dict["coord"].float(),
                        input_dict["offset"].int(),
                        input_dict["origin_coord"].float(),
                        input_dict["origin_offset"].int(),
                    )
                    pred = pred[knn_idx.flatten().long()]
                    segment = input_dict["origin_segment"]
                pred = pred.cpu().numpy()
                segment = segment.cpu().numpy()
                intersection, union, target = intersection_and_union(
                    pred, segment, self.cfg.data.num_classes[dataset_idx], self.cfg.data.ignore_index
                )
                
                # # 在主循环中，每次处理完一个批次后更新混淆矩阵
                # conf_matrix = compute_confusion_matrix(pred, segment, self.cfg.data.num_classes[dataset_idx], self.cfg.data.ignore_index)
                # confusion_matrix += conf_matrix
                
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                record[data_name] = dict(
                    intersection=intersection, union=union, target=target
                )

                mask = union != 0
                iou_class = intersection / (union + 1e-10)
                iou = np.mean(iou_class[mask])
                acc = sum(intersection) / (sum(target) + 1e-10)

                m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
                m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

                batch_time.update(time.time() - end)
                if idx % 100 == 0:
                    logger.info(
                        "Test: {} [{}/{}]-{} "
                        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                        "Accuracy {acc:.4f} ({m_acc:.4f}) "
                        "mIoU {iou:.4f} ({m_iou:.4f})".format(
                            data_name,
                            idx + 1,
                            len(test_loader_i),
                            segment.size,
                            batch_time=batch_time,
                            acc=acc,
                            m_acc=m_acc,
                            iou=iou,
                            m_iou=m_iou,
                        )
                    )

            logger.info("Syncing ...")
            comm.synchronize()
            record_sync = comm.gather(record, dst=0)

            if comm.is_main_process():
                record = {}
                for _ in range(len(record_sync)):
                    r = record_sync.pop()
                    record.update(r)
                    del r
                intersection = np.sum(
                    [meters["intersection"] for _, meters in record.items()], axis=0
                )
                union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
                target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

                iou_class = intersection / (union + 1e-10)
                accuracy_class = intersection / (target + 1e-10)
                mIoU = np.mean(iou_class)
                mAcc = np.mean(accuracy_class)
                allAcc = sum(intersection) / (sum(target) + 1e-10)

                logger.info(
                    "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                        mIoU, mAcc, allAcc
                    )
                )
                for i in range(self.cfg.data.num_classes[dataset_idx]):
                    logger.info(
                        "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[dataset_idx][i],
                            iou=iou_class[i],
                            accuracy=accuracy_class[i],
                        )
                    )
                # # 将混淆矩阵转换为LaTeX表格
                # latex_table = confusion_matrix_to_latex(confusion_matrix, self.cfg.data.names[dataset_idx])

                # # 将LaTeX表格写入日志文件
                # with open(os.path.join(self.cfg.save_path, 'confusion_matrix.log'), 'a') as f:
                #     f.write(latex_table)
                dan_weight = dan_weight.cpu().numpy() / (len(test_loader_i) * 35)
                dan_index = dan_index.cpu().numpy() / (len(test_loader_i) * 35)
                dan_weight_str = ",".join(map(str, dan_weight))
                dan_index_str = ",".join(map(str, dan_index))
                logger.info("dan_weight: {}".format(dan_weight_str))
                logger.info("dan_index: {}".format(dan_index_str))
                logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

            if len(self.cfg.data.test.condition_idx_list) > 1:
                logger.info("Set condition to condition list")
                break


@TESTERS.register_module()
class MultiSemSegSaveEvaluator(SemSegEvaluator):
    def build_test_loader(self):
        test_loader = []
        for dataset_i in self.cfg.data.test.datasets:
            test_dataset_i = build_dataset(dataset_i)
            if comm.get_world_size() > 1:
                test_sampler_i = torch.utils.data.distributed.DistributedSampler(test_dataset_i)
            else:
                test_sampler_i = None
            test_loader_i = torch.utils.data.DataLoader(
                test_dataset_i,
                batch_size=self.cfg.batch_size_test_per_gpu,
                shuffle=False,
                num_workers=self.cfg.batch_size_test_per_gpu,
                pin_memory=True,
                sampler=test_sampler_i,
                collate_fn=self.__class__.collate_fn,
            )
            test_loader.append(test_loader_i)
        return test_loader

    def test(self):
        # assert self.test_loader.batch_size == 1
        # for test_loader_i in self.test_loader:
        for test_loader_ind in range(len(self.test_loader)):
            dataset_idx = self.cfg.data.test.datasets[test_loader_ind].dataset_idx
            test_loader_i = self.test_loader[test_loader_ind]
            logger = get_root_logger()
            logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
            logger.info("Set condition to {name}".format(name=self.cfg.data.test.datasets[test_loader_ind].condition_idx))
            batch_time = AverageMeter()
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            self.model.eval()

            save_path = os.path.join(self.cfg.save_path, "result")
            make_dirs(save_path)
            
            comm.synchronize()
            record = {}
            # fragment inference
            for idx, data_dict in enumerate(test_loader_i):
                end = time.time()
                data_name = str(idx)
                input_dict = data_dict
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                if len(self.cfg.data.test.condition_idx_list) > 1:
                    output_dict_list = []
                    for condition_idx in self.cfg.data.test.condition_idx_list:
                        input_dict["condition"] = torch.zeros_like(input_dict["condition"]) + condition_idx
                        with torch.no_grad():
                            output_dict = self.model(input_dict)
                            output_dict_list.append(output_dict)
                    output = output_dict_list[0]["seg_logits"]
                    loss = output_dict_list[0]["loss"]
                    # pred = output.max(1)[1]
                    for condition_idx in range(1, len(self.cfg.data.test.condition_idx_list)):
                        loss = loss + output_dict_list[condition_idx]["loss"]
                        output = output + output_dict_list[condition_idx]["seg_logits"]
                    loss = loss / len(self.cfg.data.test.condition_idx_list)
                    pred = output.max(1)[1]
                else:
                    with torch.no_grad():
                        output_dict = self.model(input_dict)
                    output = output_dict["seg_logits"]
                    loss = output_dict["loss"]
                    pred = output.max(1)[1]

                segment = input_dict["segment"]
                if "origin_coord" in input_dict.keys():
                    knn_idx, _ = pointops.knn_query(
                        1,
                        input_dict["coord"].float(),
                        input_dict["offset"].int(),
                        input_dict["origin_coord"].float(),
                        input_dict["origin_offset"].int(),
                    )
                    pred = pred[knn_idx.flatten().long()]
                    segment = input_dict["origin_segment"]
                pred = pred.cpu().numpy()
                segment = segment.cpu().numpy()

                np.savetxt(
                    os.path.join(save_path, "{}.txt".format(idx)),
                    pred.reshape([-1, 1]),
                    fmt="%d",
                )

                intersection, union, target = intersection_and_union(
                    pred, segment, self.cfg.data.num_classes[dataset_idx], self.cfg.data.ignore_index
                )
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                record[data_name] = dict(
                    intersection=intersection, union=union, target=target
                )

                mask = union != 0
                iou_class = intersection / (union + 1e-10)
                iou = np.mean(iou_class[mask])
                acc = sum(intersection) / (sum(target) + 1e-10)

                m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
                m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

                batch_time.update(time.time() - end)
                if idx % 100 == 0:
                    logger.info(
                        "Test: {} [{}/{}]-{} "
                        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                        "Accuracy {acc:.4f} ({m_acc:.4f}) "
                        "mIoU {iou:.4f} ({m_iou:.4f})".format(
                            data_name,
                            idx + 1,
                            len(test_loader_i),
                            segment.size,
                            batch_time=batch_time,
                            acc=acc,
                            m_acc=m_acc,
                            iou=iou,
                            m_iou=m_iou,
                        )
                    )

            logger.info("Syncing ...")
            comm.synchronize()
            record_sync = comm.gather(record, dst=0)

            if comm.is_main_process():
                record = {}
                for _ in range(len(record_sync)):
                    r = record_sync.pop()
                    record.update(r)
                    del r
                intersection = np.sum(
                    [meters["intersection"] for _, meters in record.items()], axis=0
                )
                union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
                target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

                iou_class = intersection / (union + 1e-10)
                accuracy_class = intersection / (target + 1e-10)
                mIoU = np.mean(iou_class)
                mAcc = np.mean(accuracy_class)
                allAcc = sum(intersection) / (sum(target) + 1e-10)

                logger.info(
                    "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                        mIoU, mAcc, allAcc
                    )
                )
                for i in range(self.cfg.data.num_classes[dataset_idx]):
                    logger.info(
                        "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[dataset_idx][i],
                            iou=iou_class[i],
                            accuracy=accuracy_class[i],
                        )
                    )
                logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

            if len(self.cfg.data.test.condition_idx_list) > 1:
                logger.info("Set condition to condition list")
                break


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)