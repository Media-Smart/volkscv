## Introduction
This is an open source toolbox of detection based metrics based on Python and Numpy.

## Features
- The toolbox contains most commonly used metrics in detection tasks in model training 
  and evaluation, ap50, ap75, map, pr curve, etc.
  
- The metrics can be called in model evaluation period.
  
- The architecture is based on Python and Numpy, so the toolbox is not limited to the 
  framework like pytorch or tensorflow, etc, can be used in model training or evaluation.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation

### Requirements
- Linux
- Python >= 3
- Numpy >= 1.13.3

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python 3.7.3
- Numpy 1.16.4

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

>>> miou_epoch
{'mIoU': 0.8}

```

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got and modified much code from [cocoapi](https://github.com/cocodataset/cocoapi), thanks to [COCO](https://github.com/cocodataset).
