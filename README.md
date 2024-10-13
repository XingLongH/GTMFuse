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

### TNO Datasset
#### Qualitative result
<p align="center"> <img src="Fig/tno.png" width="90%"> </p>
- Four representative images of the TNO test set.In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

#### Quantitative Results

|   **Methods**    |   **EN**   |   **SD**   | **SF** | **VIF** | **AG** | **Qabf** |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:---------:|
| **GTF**         |   6.60008     | 8.69847    |  0.04613     |   0.49451      |  4.36880     |  0.43436   |
|  **FusionGAN**  |   6.50420     |   8.31568  |  0.03139    |   0.61350      | 3.20322   |  0.25814    |
| **SDNet**       |   6.58356      | 8.58165    |   0.05916     |  0.63884       | 6.03142    | 0.44332       |
| **RFN–Nest**    |  6.67323    |  8.80744  |  0.03991      |   0.67467        | 4.11122        |   0.43302   |
| **U2Fusion**    |  6.90395      | 8.98294     | 0.05854     | 0.68953       |  6.27575     |  0.45158     |
|   **LRRNet**    |6.70679       |  9.14174   | 0.05434        |0.74519         |4.82270      |  0.38131     |
|  **SwinFusion** |  6.69018     |  8.74623  | 0.04868       |  0.81244       |   4.74248     |  0.52084    |
|   **CDDFuse**   |  7.02021      |   8.92531  |   0.06416|    0.82307     |  6.07101     |  0.51846      |
| **DATFuse**     |   6.77604    |   8.82027   | 0.04612     |   0.82066     | 4.51106      |  0.51235     |
| **GTMFuse**     | **7.03991**  |**9.22010** | **0.06607**       |   **0.84018**  | **6.65676**  | **0.60472**|


###  RoadScene Datasset
#### Qualitative result
<p align="center"> <img src="Fig/road.png" width="90%"> </p>
- Four representative images of the RoadScene test set.In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

#### Quantitative Results

|   **Methods**    |   **EN**   |   **SD**   | **SF** | **VIF** | **AG** | **Qabf** |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:---------:|
| **GTF**         |   7.45805    | 10.4952   |   0.04605    |    0.57953     |   4.04458     |   0.37079    |
|  **FusionGAN**  |   7.09511    |   10.0518 |   0.04323    |   0.56307      |     4.11028   |   0.28132    |
| **SDNet**       |   7.3388     | 10.1153   |   0.07541     |   0.74513      |   7.55126    | 0.51691      |
| **RFN–Nest**    |  7.34281    |  10.2000  |  0.05192     |   0.75382       | 5.13552       |    0.45230   |
| **U2Fusion**    |  7.21249      | 10.1205    |  0.07469     |  0.67670       |  7.42630     |  0.51831     |
|   **LRRNet**    |7.09023       |   10.1468  | 0.06907       | 0.64912        | 6.19723      |  0.41013     |
|  **SwinFusion** |  7.18569     |  10.3193  | 0.06757       |  0.80244       |   6.52487     |  0.57124     |
|   **CDDFuse**   |  **7.48812**      |   **10.6921**  |   **0.09099**|     0.78466     |  **8.33022**      |  0.49671      |
| **DATFuse**     |   6.89646   |   10.4078  | 0.05495      |    0.79045     | 5.06397     |  0.50003     |
| **GTMFuse**     | 7.35795  |10.5113| 0.08181      |   **0.87918**  | 7.92432   | **0.60665**  |

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

If you have any questions, please contact me by email (hux18943@gmail.com).
