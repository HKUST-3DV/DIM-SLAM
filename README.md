# DIM-SLAM
This is official repo of **DIM-SLAM** for ICLR 2023 Paper:

 *"DENSE RGB SLAM WITH NEURAL IMPLICIT MAPS"*

Heng Li, Xiaodong Gu, Weihao Yuan, Luwei Yang, Zilong Dong, Ping Tan

[paper](https://openreview.net/pdf?id=QUK1ExlbbA), [openreview](https://openreview.net/forum?id=QUK1ExlbbA), [Project](https://poptree.github.io/DIM-SLAM/)


The core part of sfm is released. The whole framework will be made public soom.


# Implementation Detail

To ensure reproducibility, we have re-implemented the core part of the code based on the details provided in our paper. Most of the code is derived from NICE-SLAM. In the current version, DIM-SLAM includes the initialization part mentioned in the paper, and you can easily extend it to the whole sequence by adding the keyframe graph management mechanism yourself.

We are planning to release the complete code for DIM-SLAM soon. We would like to thank NICE-SLAM for providing the code for visualization.

# Bugs

Regarding the issue of slower convergence of the initial depth and pose, we have found that it takes more iterations (~1500 iter) in the current implementation compared to our original implementation (~200 iter) to achieve convergence. We are working to find the misalignment and fix this issue.

# Uasge

```
python run.py configs configs/office0.yaml
```

This reimplementation follow the configuration provided by NICE-SLAM. You could run other sequence by modifiying the setting.

# Commom Q&A

1. Q: Where can I find the video demo for your framework?

    A: The video demo can be found on the OpenReview website. For more detailed information about our framework, please refer to our paper and the OpenReview page.

2. Q: Why are the poses of the first two frames in the ground truth pose?

    A: The ground truth poses are used to fix the scale in the world coordinate system. Without doing so, the size of the grid would be meaningless. If we did not use the ground truth poses, the estimated pose would only be up to a scale.

3. Q: Does the SfM part of your framework work with the LLFF dataset?

    A: Yes, our framework works with the LLFF dataset. As shown in Figure 1, the orange and purple lines represent the BARF and our method's loss on the LLFF:Horns dataset. Our method converges much faster than BARF and achieves comparable results with the recent state-of-the-art methods as of March 2023.

4. Q: What is the time complexity of your implementation?

    A: The time complexity reported in the paper is based on our fully CUDA implementation. This repository only contains the PyTorch version for research purposes.

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

