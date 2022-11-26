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
| YOLOv3<br><sup>(<u>Paper:page_with_curl:</u>)</br> | COCO | trainval | test-dev | 416 | 31.2 | 55.3 | 61.95 | 65.86 |
| YOLOv3-416<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 416 | - | - | 61.55 | 65.60 |


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
2022-11-16 13:45:37 | YOLOv3 Architecture Info - Params(M): 61.55, FLOPS(B): 65.60
2022-11-16 13:48:09 | [Train-Epoch:001] multipart: 308.6470  obj: 1.8051  noobj: 380.8712  txty: 0.8276  twth: 20.2762  cls: 10.8872  
2022-11-09 18:01:53 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.336
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.054
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.011
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.131
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.405

                                                ...

2022-11-09 22:59:17 | [Train-Epoch:135] multipart: 1.6719  obj: 0.3912  noobj: 0.3355  box: 0.1720  cls: 0.2530  
2022-11-09 22:59:40 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.292
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.036
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.113
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.293
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.399
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.688
```


<div align="center">

<a href=""><img src=./asset/car.jpg width="22%" /></a> <a href=""><img src=./asset/cat.jpg width="22%" /></a> <a href=""><img src=./asset/dog.jpg width="22%" /></a> <a href=""><img src=./asset/person.jpg width="22%" /></a>

</div>


---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  