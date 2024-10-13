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
#### Qualitative result
<p align="center"> <img src="Fig/msrs.png" width="90%"> </p>
- Four representative images of the MSRS test set. In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

#### Quantitative Results

|   **Methods**    |   **EN**   |   **SD**   | **SF** | **VIF** | **AG** | **Qabf** |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:-------------------------:|
| **GTF** |   4.44195   | 6.11111  |   0.05620    |    0.48176     |           3.53966            |           0.39194           |
|  **FusionGAN**   |   5.86785   |   6.79263    |   0.03654    |    0.61998      |           3.00051            |           0.24709            |
| **SDNet**  |   5.54468    |   6.13925    |   0.05910    |    0.48644      |        4.40836          |         0.41903           |
| **RFN–Nest**  |   5.81096    |   7.91701    |   0.04982    |   0.74520     |          4.12198           |           0.50474             |
| **U2Fusion**  |  5.03625    | 6.78870  |   0.06157    |   0.57216     |          4.48894            |           0.42512             |
|   **LRRNet**    |5.89799 |   7.30930    | 0.04548  |  0.38422   |          3.64204          |          0.19980           |
|  **SwinFusion**  |  6.61543    |   8.46817   | 0.06756     |  0.99403     |         5.26562           |        0.66481            |
|   **CDDFuse**   |  6.32740   |   8.53021    |   **0.08130**  |     0.97155       |       6.12164           |         0.66558           |
| **DATFuse**  |    6.29844    |   7.71886      | 0.07247  |     0.71196     |       5.96856          |       0.54618          |
| **GTMFuse** |   **6.78256**  |  **8.60603**   | 0.08105  |   **1.00857**   |        **6.39748**          |      **0.69590**        |

###  RoadScene Datasset

<p align="center"> <img src="Fig/road.png" width="90%"> </p>
- Four representative images of the RoadScene test set.

### TNO Datasset

<p align="center"> <img src="Fig/tno.png" width="90%"> </p>
- Four representative images of the RoadScene test set.

If you have any questions, please contact me by email (hux18943@gmail.com).
