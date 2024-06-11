# Improving Generalization Ability for 3D Object Detection by Learning Sparsity-invariant Features 
The following figure is the overall structure of our method.
![image](./figures/structure.png)

## Introduction
In this paper, we propose a training method to improve the generalization ability for 3D object detection on a single domain. Our method empowers the 3D detector to learn sparsity-invariant features through training with our proposed augmentation and feature alignment techniques.
![image](./figures/fig1.png)

## Model Zoo

<!-- ### General models

General models trained on Waymo dataset. The training Waymo data used in our work is version 1.0.

### Single-dataset Generalization

Results of single-dataset generalization on KITTI dataset with SECOND-IoU (moderate difficulty). -->

You can download all the pretrained models in the following [link](https://drive.google.com/file/d/1dE-uBtGcD8EpoYxd1vOCFe6WC1GaIm3x/view?usp=drive_link).


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md).

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md).

## License

Our code is released under the Apache 2.0 license.

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.2](https://github.com/open-mmlab/OpenPCDet/tree/v0.2.0) and [ST3D](https://github.com/CVMI-Lab/ST3D). Thanks OpenPCDet Development Team for their awesome codebase.

## Citation

If you find this project useful in your research, please consider cite:
```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```
