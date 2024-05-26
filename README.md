# GTMFuse: Group-attention transformer-driven multiscale dense feature-enhanced network for infrared and visible image fusion

[Paper link](https://doi.org/10.1016/j.knosys.2024.111658)

## To Train 
 ```
python train.py 
```

## To Test
1. Downloading the pre-trained checkpoint from [best_model.pth](https://pan.baidu.com/s/1KRxEpTCM0t4fPgvPz9iRaQ?pwd=of0u) and putting it in **./checkpoints**.
2. python test.py

## HBUT dataset
Downloading the HBUT dataset from [HBUT](https://pan.baidu.com/s/1AcPukklhBTSL3SOJZC2D2Q?pwd=31ys)

## If this work is helpful to you, please cite it as：
```
@article{mei2024gtmfuse,
  title={GTMFuse: Group-attention transformer-driven multiscale dense feature-enhanced network for infrared and visible image fusion},
  author={Mei, Liye and Hu, Xinglong and Ye, Zhaoyi and Tang, Linfeng and Wang, Ying and Li, Di and Liu, Yan and Hao, Xin and Lei, Cheng and Xu, Chuan and others},
  journal={Knowledge-Based Systems},
  volume={293},
  pages={111658},
  year={2024},
  publisher={Elsevier}
}
```

# HATNet
* The pytorch implementation for HATNet in paper "Hybrid Attention-aware Transformer Network Collaborative Multiscale Feature Alignment for Building Change Detection".

# Requirements
* Python 3.6
* Pytorch 1.7.0

# Datasets Preparation
The path list in the datasest folder is as follows:

|—train

* ||—A

* ||—B

* ||—OUT

|—val

* ||—A

* ||—B

* ||—OUT

|—test

* ||—A

* ||—B

* ||—OUT


where A contains pre-temporal images, B contains post-temporal images, and OUT contains ground truth images.
# Train
* python train.py --dataset-dir dataset-path
# Test
* python eval.py --ckp-paths weight-path --dataset-dir dataset-path
# Visualization
* python visualization visualization.py --ckp-paths weight-path --dataset-dir dataset-path (Note that batch-size must be 1 when using visualization.py)
* Besides, you can adjust the parameter of full_to_color to change the color

# Citation
If this work is helpful to you, please cite it as:
```
@article{xu2024hybrid,
  title={Hybrid Attention-Aware Transformer Network Collaborative Multiscale Feature Alignment for Building Change Detection},
  author={Xu, Chuan and Ye, Zhaoyi and Mei, Liye and Yu, Haonan and Liu, Jianchen and Yalikun, Yaxiaer and Jin, Shuangtong and Liu, Sheng and Yang, Wei and Lei, Cheng},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={73},
  pages={1--14},
  year={2024},
  publisher={IEEE}
}
```
