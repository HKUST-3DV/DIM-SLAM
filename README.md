# DIM-SLAM
This is official repo for ICLR 2023 Paper "DENSE RGB SLAM WITH NEURAL IMPLICIT MAPS"
Heng Li, Xiaodong Gu, Weihao Yuan, Luwei Yang, Zilong Dong, Ping Tan

[paper](https://openreview.net/pdf?id=QUK1ExlbbA), [openreview](https://openreview.net/forum?id=QUK1ExlbbA), [Project](https://poptree.github.io/DIM-SLAM/)


The core part of sfm is released. The whole framework will be made public soom.

# Commom Q&A

1. Q: Why the poses of first two frames are GT pose?
A: To fix the scale on world coordinate, otherwise the size of the grid is meaningless. If the first, the estimated pose will up to a scale.

2. Q: Does the SfM part works on LLFF?
A: Yes. As shown in Fig 1, the orange/purple line are the loss of BARF/ours on LLFF:Horns, ours method converage much faster than BARF. Our method also has comparable result with recent SOTA methods till Mar. 2023. 

![Fig1: LLFF:Horns, Ours compares with BARF](./figs/20230330193005.png)


If you find our code or paper useful, please cite:
```
@inproceedings{li2023dense,
  author    = {Li, Heng and Gu, Xiaodong and Yuan, Weihao and Yang, Luwei and Dong, Zilong and Tan, Ping},
  title     = {Dense RGB SLAM With Neural Implicit Maps},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year      = {2023},
  url={https://openreview.net/forum?id=QUK1ExlbbA}
}
```

