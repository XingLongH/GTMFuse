# GTMFuse: Group-attention transformer-driven multiscale dense feature-enhanced network for infrared and visible image fusion
⭐ This code has been completely released ⭐ 

⭐ our [article](https://doi.org/10.1016/j.knosys.2024.111658) ⭐ 

If our code is helpful to you, please cite:

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
## To Train 
 ```
python train.py 
```

## To Test
1. Downloading the pre-trained checkpoint from [best_model.pth](https://pan.baidu.com/s/1KRxEpTCM0t4fPgvPz9iRaQ?pwd=of0u) and putting it in **./checkpoints**.
2. python test.py

## HBUT dataset
Downloading the HBUT dataset from [HBUT](https://pan.baidu.com/s/1AcPukklhBTSL3SOJZC2D2Q?pwd=31ys)

## overall network

<p align="center"> <img src="Fig/overall network.png" width="90%"> </p>

## Results

### MSRS Dataset

<p align="center"> <img src="Fig/msrs.png" width="90%"> </p>
- Four representative images of the MSRS test set. In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

###  RoadScene Datasset

<p align="center"> <img src="Fig/road.png" width="90%"> </p>
- Four representative images of the RoadScene test set.

### TNO Datasset

<p align="center"> <img src="Fig/tno.png" width="90%"> </p>
- Four representative images of the RoadScene test set.

If you have any questions, please contact me by email (hux18943@gmail.com).
