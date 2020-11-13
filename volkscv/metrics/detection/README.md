## Support
- [x] [AveragePrecision](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/average_precision.py)
- [x] [AverageRecall](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/average_recall.py)
- [x] [PRCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/pr_curve.py)
- [x] [SupercatePRCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/pr_curve.py)

## Usage
### Known Issues
- this metric is based on pycocotools, so the inputs (pred , target, etc.) of all metrics have to follow COCO format.
- For 'file' mode:
- pred: [{'image_id': XX, 'bbox': [x, y, w, h], 'score': X, 'category_id': X }, ...]
- target: {'info': {},
           'licenses': [],
           'images': [
              {'file_name': X,
               'height': X,
               'width': X,
               'id': X
              },
              ...
           ],
           'annotations':[
              {'segmentation': [],
               'area': X,
               'iscrowd': 0 or 1,
               'image_id': X,
               'bbox': [x, y w, h],
               'category_id': X,
               'id': X,
               },
               ...
           ],
           'categories':[
              {'id': X,
               'name': X,
               'supercategory': X,
              },
              ...
           ]
          }
- For 'online' mode:
- pred: list.
    Prediction results from detection model, stored in a list, with length of B = batchsize,
    each element of the list is a numpy.ndarray with shape of (N, 6),
    N is the number of bboxes, 6 stands for the information of box [x, y, w, h, score, cat_id]
- target: list.
    Ground truth, stored in a list, with length of B = batchsize, each element of the list is
    a numpy.ndarray with shape of (M, 6), M is the number of bboxes, 6 stands for the information
    of box [x, y, w, h, cat_id, iscrowd]
- 'online' mode requires easier format for computing metrics, which is easier assert in training 
    process to calculate metrics only for current batch, while 'file' mode usually used after 
    test process when results are stored in files (offline).
- the output value of all metrics are stored in a dict format.

Take ap30, ap50, map for example in pytorch detection evaluation:

Using 'online' mode:

```shell
from volkscv.metrics.detection import AveragePrecision

for epoch in range(num_max_epoch):
    ap = AveragePrecision(iou=[0.3, 0.5], mode='file')
    for (image, target) in dataloader:
        ...
        pred = model(image)
        loss = ...
        ...
        # calculate ap of the batch
        ap_batch = ap(pred, target)
    # calculate ap of the epoch
    ap_epoch = ap.accumulate()

>>> ap_batch
 Average Precision  (AP) @[ IoU=0.30:0.50 | area=   all | maxDets=100 ] = 0.952
 Average Precision  (AP) @[ IoU=0.30:0.50 | area= small | maxDets=100 ] = 0.835
 Average Precision  (AP) @[ IoU=0.30:0.50 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.30:0.50 | area= large | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.30      | area=   all | maxDets=100 ] = 0.964
 Average Precision  (AP) @[ IoU=0.30      | area= small | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.30      | area=medium | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.30      | area= large | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.941
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.800
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 1.000
{'map': [0.9522637817396231, 0.8352953014125928, 0.466828378060557, 0.9999339763655164], 
 'ap': [0.9639679068730561, 0.8710708070160018, 0.47649516834160294, 0.9999834431603696, 0.9405596566061901, 0.7995197958091839, 0.45716158777951105, 0.9998845095706632]}

>>> ap_epoch
 Average Precision  (AP) @[ IoU=0.30:0.50 | area=   all | maxDets=100 ] = 0.952
 Average Precision  (AP) @[ IoU=0.30:0.50 | area= small | maxDets=100 ] = 0.835
 Average Precision  (AP) @[ IoU=0.30:0.50 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.30:0.50 | area= large | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.30      | area=   all | maxDets=100 ] = 0.964
 Average Precision  (AP) @[ IoU=0.30      | area= small | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.30      | area=medium | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.30      | area= large | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.941
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.800
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 1.000
{'map': [0.9522637817396231, 0.8352953014125928, 0.466828378060557, 0.9999339763655164], 
 'ap': [0.9639679068730561, 0.8710708070160018, 0.47649516834160294, 0.9999834431603696, 0.9405596566061901, 0.7995197958091839, 0.45716158777951105, 0.9998845095706632]}

```

Using 'file' mode:

```shell
from volkscv.metrics.detection import AveragePrecision

for epoch in range(num_max_epoch):
    ap = AveragePrecision(iou=[0.3, 0.5], mode='file')
    for (image, target) in dataloader:
        ...
        pred_output = model(image)
        loss = ...
        ...
    # calculate ap of the epoch
    ap_epoch = ap(pred, target)

>>> ap_epoch
 Average Precision  (AP) @[ IoU=0.30:0.50 | area=   all | maxDets=100 ] = 0.952
 Average Precision  (AP) @[ IoU=0.30:0.50 | area= small | maxDets=100 ] = 0.835
 Average Precision  (AP) @[ IoU=0.30:0.50 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.30:0.50 | area= large | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.30      | area=   all | maxDets=100 ] = 0.964
 Average Precision  (AP) @[ IoU=0.30      | area= small | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.30      | area=medium | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.30      | area= large | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.941
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.800
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 1.000

{'map': [0.9522637817396231, 0.8352953014125928, 0.466828378060557, 0.9999339763655164], 
 'ap': [0.9639679068730561, 0.8710708070160018, 0.47649516834160294, 0.9999834431603696, 0.9405596566061901, 0.7995197958091839, 0.45716158777951105, 0.9998845095706632]}

```

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got and modified much code from [cocoapi](https://github.com/cocodataset/cocoapi), thanks to [COCO](https://github.com/cocodataset).
