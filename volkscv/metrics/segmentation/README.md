## Introduction
This is an open source toolbox of segmentation based metrics based on Python and Numpy.

## Features
- The toolbox contains most commonly used metrics in segmentation tasks in model training 
  and evaluation, including confusion matrix, accuracy, IoU, mIoU, dice score, etc.
  
- The metrics can be called in model training process or model evaluation period.
  
- The architecture is based on Python and Numpy, so the toolbox is not limited to the 
  framework like pytorch or tensorflow, etc, used in model training or evaluation.

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
- the inputs (pred , target, etc.) of all metrics have to be numpy array.
- the output value of all metrics are stored in a dict format.
 
Take miou for example in pytorch segmentation training:

```shell
from volkscv.metrics.segmentation import mIoU

for epoch in range(num_max_epoch):
    miou = mIoU(num_classes=10)
    for (image, target) in dataloader:
        ...
        pred = model(image)
        loss = ...
        ...
        # calculate miou for current batch
        miou_current_batch = miou(pred.numpy(), target.numpy())
        # calculate miou from the start of current epoch till current batch
        miou_current_average = miou.accumulate()
    # calculate miou of the epoch
    miou_epoch = miou.accumulate()

>>> miou_epoch
{'mIoU': 0.8}

```

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).
