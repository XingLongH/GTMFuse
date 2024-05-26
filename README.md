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

# SCFNet: Lightweight Steel Defect Detection Network Based on Spatial Channel Reorganization and Weighted Jump Fusion
⭐ This code has been completely released ⭐ 

⭐ our [article](https://www.mdpi.com/2227-9717/12/5/931) ⭐ 

If our code is helpful to you, please cite:

```
@article{li2024scfnet,
  title={SCFNet: Lightweight Steel Defect Detection Network Based on Spatial Channel Reorganization and Weighted Jump Fusion},
  author={Li, Hongli and Yi, Zhiqi and Mei, Liye and Duan, Jia and Sun, Kaimin and Li, Mengcheng and Yang, Wei and Wang, Ying},
  journal={Processes},
  volume={12},
  number={5},
  pages={931},
  year={2024},
  publisher={MDPI}
}
```


[//]: # (* [**Requirements**]&#40;#Requirements&#41;)

[//]: # (* [**Train**]&#40;#Train&#41;)

[//]: # (* [**Test**]&#40;#Test&#41;)

[//]: # (* [**Results**]&#40;#Results&#41;)

[//]: # (* [**Time**]&#40;#Time&#41;)

[//]: # (* [**Visualization of results**]&#40;#Visualization-of-results&#41;)

[//]: # (* [**Acknowledgements**]&#40;#Acknowledgements&#41;)

[//]: # (* [**Contact**]&#40;#Contact&#41;)




## Requirements

```python
pip install -r requirements.txt
```


<p align="center"> <img src="Fig/SCFNet.png" width="80%"> </p>


## Train

### 1. Prepare training data 

- The download link for the NEU-DET data set is [here](https://pan.baidu.com/s/1iADDCBTB6r4OaxlOPRJsMQ?pwd=gmu2).
- The download link for the GC10-DET data set is [here](https://pan.baidu.com/s/1Eg7pbKJVlBQ698v9B1oMsw?pwd=a9zr).
```python
SCFNet
├── NEU-DET
│   ├── images
│   │   ├── train
│   │   │   ├── crazing_1.jpg
│   │   │   ├── crazing_2.jpg
│   │   │   ├── .....
│   │   ├── val
│   │   ├── test
│   ├── labels
│   │   ├── train
│   │   │   ├── crazing_1.txt
│   │   │   ├── crazing_2.txt
│   │   │   ├── .....
│   │   ├── val
│   │   ├── test
```
- After downloading the data set, modify the paths in path, train, val and test in the [data.yaml](data.yaml) file.


### 2. Begin to train
```python
python train.py
```


## Test

### 1. Begin to test
```python
python val.py
```

## Results

|   **Methods**    |   **P**   |   **R**   | **mAP50** | **mAP50:95** | **GFLOPs** $\downarrow$ | **Params/M** $\downarrow$ |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:-------------------------:|
| **Faster R-CNN** |   0.610   | **0.865** |   0.76    |    0.377     |           135           |           41.75           |
|  **CenterNet**   |   0.712   |   0.749   |   0.764   |    0.412     |           123           |           32.12           |
| **YOLOv5n-7.0**  |   0.694   |   0.694   |   0.746   |    0.422     |         **4.2**         |         **1.77**          |
| **YOLOv5s-7.0**  |   0.745   |   0.719   |   0.761   |    0.429     |          15.8           |           7.03            |
| **YOLOv7-tiny**  |   0.645   | **0.775** |   0.753   |    0.399     |          13.1           |           6.02            |
|   **YOLOv8s**    | **0.768** |   0.726   | **0.795** |  **0.467**   |          28.4           |           11.13           |
|  **YOLOX-tiny**  |   0.746   |   0.768   |   0.76    |    0.357     |          7.58           |           5.03            |
|   **MRF-YOLO**   |   0.761   |   0.707   |   0.768   |      -       |          29.7           |           14.9            |
| **YOLOv5s-FCC**  |     -     |     -     | **0.795** |      -       |            -            |           13.35           |
| **WFRE-YOLOv8s** |   0.759   |   0.736   |   0.794   |    0.425     |          32.6           |           13.78           |
|    **CG-Net**    |   0.734   |   0.687   |   0.759   |    0.399     |           6.5           |            2.3            |
|   **ACD-YOLO**   |     -     |     -     |   0.793   |      -       |          21.3           |             -             |
|  **YOLOv5-ESS**  |     -     |   0.764   |   0.788   |      -       |            -            |           7.07            |
|  **PMSA-DyTr**   |     -     |     -     | **0.812** |      -       |            -            |             -             |
|   **MED-YOLO**   |     -     |     -     |   0.731   |    0.376     |           18            |           9.54            |
|   **MAR-YOLO**   |     -     |     -     |   0.785   |      -       |          20.1           |             -             |
|    **SCFNet**    | **0.786** |   0.715   | **0.812** |  **0.469**   |         **5.9**         |           **2**           |

- Bold indicates first or second best performance.
## Time
2024.4.25 open the val.py

2024.5.16 update train.py

2024.5.16 update ScConv module.


<p align="center"> <img src="Fig/ScConv.png" width="80%"> </p>

2024.5.25 update readme.

## Visualization of results


<p align="center"> <img src="Fig/results.png" width="80%"> </p>


## Acknowledgements
This code is built on [ultralytics (PyTorch)](https://github.com/ultralytics/ultralytics). We thank the authors for sharing the codes.

[//]: # (## Licencing)

[//]: # ()
[//]: # (Copyright &#40;C&#41; 2024 Zhiqi Yi)

[//]: # ()
[//]: # (This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.)

[//]: # ()
[//]: # (This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.)

[//]: # ()
[//]: # (You should have received a copy of the GNU General Public License along with this program.)

## Contact
If you have any questions, please contact me by email (lazyshark2001@gmail.com).



