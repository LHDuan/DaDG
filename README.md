# DaDG: Distribution-aware Multi-source Domain Generalization for Point Cloud Segmentation

This repository contains the official PyTorch implementation for the paper:

**Distribution-aware Multi-source Domain Generalization for Point Cloud Segmentation**

Our work addresses the task of multi-source domain generalization for point cloud segmentation with two key modules:
- **Distribution-aware Normalization (DAN):** Captures domain-specific feature statistics (mean and variance) from multiple sources and performs inference-time generalization by dynamically selecting these statistics for unseen target data.
- **Dual-label Alignment (DLA):** Preserves rich, fine-grained semantic information from source domains while learning a shared, coarse-grained label space for evaluation. This is achieved by simultaneously aligning point cloud features with text embeddings of both label types.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/LHDuan/DaDG.git
    cd DaDG
    ```

2.  **Create Conda Environment:** We recommend using Anaconda or Miniconda to manage dependencies.
    ```bash
    conda create -n dadg python=3.8 -y
    conda activate dadg
    ```

3.  **Install Dependencies:** Our codebase builds upon [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3/). Please follow their installation instructions for core dependencies like PyTorch and CUDA. Our development environment used the following versions:
    - Python: `3.8.19`
    - PyTorch: `2.0.1`
    - CUDA Toolkit: `11.7`

## Data Preparation

1.  **Indoor Dataset:**
Please download [3D-Front](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/ETXTrSJmy8lLikn0I_zsOisB5utQnffuqp3dGYwv-IIzDw?e=tpHJqc), [S3DIS](https://huggingface.co/datasets/Pointcept/s3dis-compressed), and [ScanNet](https://huggingface.co/datasets/Pointcept/scannet-compressed) datasets, and link the processed dataset directory to the codebase.

2.  **Outdoor Dataset:**
Please download SemanticKITTI, nuScenes, and Waymo datasets follonwing the instructions in [Pointcept](https://github.com/Pointcept/Pointcept), and the [SynLiDAR](https://github.com/xiaoaoran/SynLiDAR) dataset.

    The structure of the data folder should be:

    ```
    ./data
    ├── 3d_front/
    │   ├── train/
    │   └── val/
    │
    ├── scannet/
    │   ├── train/
    │   └── val/
    │
    ├── s3dis/
    │   ├── Area_1/
    │   ├── Area_2/
    │   └── ...
    │
    ├── nuscenes/
    │   ├── info/
    │   ├── lidarseg/
    │   └── samples/
    │
    ├── semantic_kitti/
    │   └── dataset/
    │       └── sequences/
    │           ├── 00/
    │           ├── 01/
    │           └── ...
    │
    ├── synlidar/
    │   ├── 00/
    │   ├── 01/
    │   └── ...
    │
    └── waymo/
        ├── training/
        └── validation/
    ```

## Training Pipeline
To train a model from scratch, use the `train.sh` script.

**Indoor Example:**
```bash
# Example for 3DFront + S3DIS −→ ScanNet setting (modify GPU IDs as needed)
# Checkpoints will be saved under `./exp/ft_s3_2_sc/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d ft_s3_2_sc -c dadg-ptv3-bn -n dadg-ptv3-bn

# Example for 3DFront + ScanNet −→ S3DIS setting (modify GPU IDs as needed)
# Checkpoints will be saved under `./exp/ft_sc_2_s3/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d ft_sc_2_s3 -c dadg-ptv3-bn -n dadg-ptv3-bn
```

**Outdoor Example:**

```bash
# Example for SynLiDAR + SemanticKITTI −→ nuScenes + Waymo setting (modify GPU IDs as needed)
# Checkpoints will be saved under `./exp/synlidar_sk_2_nu_wa/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d synlidar_sk_2_nu_wa -c dadg-ptv3-bn -n dadg-ptv3-bn

# Example for SynLiDAR + Waymo −→ nuScenes + SemanticKITTI setting (modify GPU IDs as needed)
# Checkpoints will be saved under `./exp/synlidar_wa_2_nu_sk/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d synlidar_wa_2_nu_sk -c dadg-ptv3-bn -n dadg-ptv3-bn
```

## Evaluation
To evaluate a trained model, use the `test.sh` script.

**Indoor Example:**

```bash
# Example for 3DFront + S3DIS −→ ScanNet setting (modify GPU IDs as needed)
# The testing log file will be save under `./exp/ft_s3_2_sc/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0 sh scripts/test.sh -g 1 -d ft_s3_2_sc -c dadg-ptv3-bn-eval -n dadg-ptv3-bn -w model_last

# Example for 3DFront + ScanNet −→ S3DIS setting (modify GPU IDs as needed)
# The testing log file will be save under `./exp/ft_sc_2_s3/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0 sh scripts/test.sh -g 1 -d ft_sc_2_s3 -c dadg-ptv3-bn-eval -n dadg-ptv3-bn -w model_last
```

**Outdoor Example:**

```bash
# Example for SynLiDAR + SemanticKITTI −→ nuScenes + Waymo setting (modify GPU IDs as needed)
# The testing log file will be save under `./exp/synlidar_sk_2_nu_wa/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0 sh scripts/test.sh -g 1 -d synlidar_sk_2_nu_wa -c dadg-ptv3-bn-eval -n dadg-ptv3-bn -w model_last

# Example for SynLiDAR + Waymo −→ nuScenes + SemanticKITTI setting (modify GPU IDs as needed)
# The testing log file will be save under `./exp/synlidar_wa_2_nu_sk/dadg-ptv3-bn/`
CUDA_VISIBLE_DEVICES=0 sh scripts/test.sh -g 1 -d synlidar_wa_2_nu_sk -c dadg-ptv3-bn-eval -n dadg-ptv3-bn -w model_last
```

## Citations
If you find our work useful in your research, please consider citing:
```
@inproceedings{duan2025dadg,
  title={Distribution-aware Multi-source Domain Generalization for Point Cloud Segmentation},
  author={Duan, Lunhao and Jiang, jiaqin and Zhao, Shanshan and and Xia, Gui-Song},
  booktitle={},
  year={2025}
}
```

## Contact
[lhduan@whu.edu.cn](lhduan@whu.edu.cn)

## Acknowledgements
Our work benefits greatly from the open-source community. We would like to thank the authors of the following projects for their excellent work:
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3/)
- [Pointcept](https://github.com/Pointcept/Pointcept)
- [DGLSS](https://github.com/gzgzys9887/DGLSS)
- [LiDOG](https://github.com/saltoricristiano/lidog)
- [SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)
- All the dataset providers.

We sincerely thank the authors for making their code publicly available.