# <div align="center">YOLOv3</div>

---

## [Content]
1. [Description](#description)   
2. [Usage](#usage)  
2-1. [K-medoids Anchor Clustering](#k-medoids-anchor-clustering)  
2-2. [Model Training](#model-training)  
2-3. [Detection Evaluation](#detection-evaluation)  
2-4. [Result Analysis](#result-analysis)  
3. [Contact](#contact)   

---

## [Description]

This is a repository for PyTorch implementation of YOLOv3 following the original paper (https://arxiv.org/abs/1804.02767).   

 - **Performance Table**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | mAP<br><sup>(@0.5:0.95) | mAP<br><sup>(@0.5) | Params<br><sup>(M) | FLOPS<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| YOLOv3<br><sup>(<u>Paper:page_with_curl:</u>)</br> | COCO | train2017 | val2017 | 416 | 31.2 | 55.3 | 61.95 | 65.86 |
| YOLOv3<br><sup>(<u>Our:star:</u>)</br> | COCO | train2017 | val2017 | 416 | - | - | 61.55 | 65.60 |
| YOLOv3<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 416 | - | - | 61.55 | 65.60 |


 - **Pretrained Model Download**

	- [DarkNet-53 Backbone](https://drive.google.com/file/d/19lyl0lROc7UuTB-qKg38_WOxX8Id7Gxq/view?usp=share_link)



<div align="center">

  <a href=""><img src=./asset/EP_150.jpg width="100%" />

</div>


## [Usage]

#### K-medoids Anchor Clustering   
 - You extract anchor box priors from all instances' boxes at first.

 ```python
python kmedoids_anchor.py --exp_name my_test --data voc.yaml  --n_cluster 9
 ```


```log
2022-11-16 13:43:54 | Avg IOU: 69.07%
2022-11-16 13:43:54 | Boxes:
    [[0.076      0.21866667]
    [0.052      0.09      ]
    [0.144      0.136     ]
    [0.314      0.27466667]
    [0.446      0.752     ]
    [0.856      0.862     ]
    [0.258      0.56      ]
    [0.65       0.45045045]
    [0.15       0.35733333]]
2022-11-16 13:43:54 | Ratios: [0.35, 0.42, 0.46, 0.58, 0.59, 0.99, 1.06, 1.14, 1.44]
```

<div align="center">

  <a href=""><img src=./asset/box_hist.jpg width="40%" /></a>

</div>


#### Model Training 
 - You can train your own YOLOv3 model using Darknet-53 with anchor box from above step.

```python
python train.py --exp_name my_test --data voc.yaml
```


#### Detection Evaluation
 - You can compute detection metric via mean Average Precision(mAP) with IoU of 0.5, 0.75, 0.5:0.95. I follow the evaluation code with the reference on https://github.com/rafaelpadilla/Object-Detection-Metrics.

```python
python val.py --exp_name my_test --data voc.yaml --ckpt_name best.pt
```


#### Result Analysis
 - After training is done, you will get the results shown below.

<div align="center">

  <a href=""><img src=./asset/figure_AP.jpg width="60%" /></a>

</div>


```log
2022-11-26 22:16:17 | YOLOv3 Architecture Info - Params(M): 61.65, FLOPS(B): 65.74
2022-11-26 22:21:53 | [Train-Epoch:001] multipart: 430.8379  obj: 0.5071  noobj: 421.2422  txty: 0.3636  twth: 1.1664  cls: 5.5302  
2022-11-26 22:27:30 | [Train-Epoch:002] multipart: 7.2063  obj: 0.7710  noobj: 0.1393  txty: 0.3232  twth: 0.4657  cls: 2.4231  
2022-11-26 22:33:07 | [Train-Epoch:003] multipart: 6.5012  obj: 0.7929  noobj: 0.2720  txty: 0.3107  twth: 0.3330  cls: 1.6208  
2022-11-26 22:38:39 | [Train-Epoch:004] multipart: 6.1957  obj: 0.7574  noobj: 0.3817  txty: 0.2998  twth: 0.3036  cls: 1.4235  
2022-11-26 22:44:14 | [Train-Epoch:005] multipart: 5.9186  obj: 0.7301  noobj: 0.4557  txty: 0.2879  twth: 0.2594  cls: 1.2649  
2022-11-26 22:49:48 | [Train-Epoch:006] multipart: 5.7983  obj: 0.7167  noobj: 0.4926  txty: 0.2797  twth: 0.2444  cls: 1.1983  
2022-11-26 22:55:27 | [Train-Epoch:007] multipart: 5.6003  obj: 0.6963  noobj: 0.5234  txty: 0.2740  twth: 0.2339  cls: 1.0875  
2022-11-26 23:01:03 | [Train-Epoch:008] multipart: 5.4977  obj: 0.6828  noobj: 0.5541  txty: 0.2669  twth: 0.2198  cls: 1.0429  
2022-11-26 23:05:44 | [Train-Epoch:009] multipart: 5.3483  obj: 0.6623  noobj: 0.6062  txty: 0.2594  twth: 0.2071  cls: 0.9639  
2022-11-26 23:11:28 | [Train-Epoch:010] multipart: 5.2561  obj: 0.6460  noobj: 0.6265  txty: 0.2553  twth: 0.1975  cls: 0.9466  
2022-11-26 23:12:54 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.270
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.075
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.165
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.075
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.177
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.363
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.664

                                                ...

2022-11-27 12:59:59 | [Train-Epoch:100] multipart: 3.2324  obj: 0.3703  noobj: 0.8265  txty: 0.1715  twth: 0.0972  cls: 0.2859  
2022-11-27 13:01:17 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.759
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.536
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.157
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.303
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.284
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.517
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.796
```


<div align="center">

<a href=""><img src=./asset/car.jpg width="22%" /></a> <a href=""><img src=./asset/cat.jpg width="22%" /></a> <a href=""><img src=./asset/dog.jpg width="22%" /></a> <a href=""><img src=./asset/person.jpg width="22%" /></a>

</div>


---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  