## Support
- [x] [AveragePrecision](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/average_precision.py)
- [x] [AverageRecall](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/average_recall.py)
- [x] [PRCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/pr_curve.py)
- [x] [SupercatePRCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/pr_curve.py)

## Usage
### Known Issues
- this metric is based on pycocotools, so the inputs (pred , target, etc.) of all metrics have to be 
json file path where stores data in COCO format.
- the output value of all metrics are stored in a dict format.

Take ap30, ap50, map for example in pytorch detection evaluation:

```shell
from volkscv.metrics.detection import AveragePrecision

for epoch in range(num_max_epoch):
    ap = AveragePrecision(iou=[0.3, 0.5])
    for (image, target) in dataloader:
        ...
        pred = model(image)
        loss = ...
        ...
        # calculate ap from the start of current epoch till current batch
        ap_current = ap(pred_file_path, target_file_path)
    # calculate ap of the epoch
    ap_epoch = ap(pred_file_path, target_file_path)

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
