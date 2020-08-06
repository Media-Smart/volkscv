## Support
- [x] [AveragePrecision](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/average_precision.py)
- [x] [AverageRecall](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/average_recall.py)
- [x] [PRCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/detection/pr_curve.py)

## Usage
### Known Issues
- this metric is based on pycocotools, so the inputs (pred , target, etc.) of all metrics have to be 
json file path where stores data in COCO format.
- the output value of all metrics are stored in a dict format.

Take ap50, map for example in pytorch detection evaluation:

```shell
from volkscv.metrics.detection import AveragePrecision

for epoch in range(num_max_epoch):
    ap = AveragePrecision(ap_mode=('map', 'ap50'))
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
{'map': 0.807, 'ap50': 0.995}

```

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got and modified much code from [cocoapi](https://github.com/cocodataset/cocoapi), thanks to [COCO](https://github.com/cocodataset).
