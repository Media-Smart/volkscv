## Introduction
This is an open source toolbox of classification based metrics based on Python and Numpy.

## Features
- The toolbox contains most commonly used metrics in classification tasks in model training 
  and evaluation, including accuracy, confusion matrix with associated analysis, F-beta score, 
  precision-recall curve, ROC curve, etc.
  
- The metrics covers tasks like binary classification, multiclass classification and 
  multilabel classification, etc, can be called in model training process or model evaluation period.
  
- The architecture is based on Python and Numpy, so the toolbox is not limited to the 
  framework like pytorch or tensorflow, etc, used in model training or evaluation.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation

### Requirements
- Linux
- Python >= 3
- Numpy >= 1.13.3
- Scipy >= 0.19.1

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- Python 3.7.3
- Numpy 1.16.4

## Usage
### Known Issues
- the inputs (pred , target, etc.) of all metrics have to be numpy array.
- the output value of all metrics are stored in a dict format.
 
Take topk accuracy for example in pytorch classification training:

```shell
from volkscv.metrics.classification import TopKAccuracy

for epoch in range(num_max_epoch):
    topkacc = TopKAccuracy(topk=(1,3,5))
    for (image, target) in dataloader:
        ...
        pred = model(image)
        loss = ...
        ...
        # calculate acc for current batch
        acc_current_batch = topkacc(pred.numpy(), target.numpy())
        # calculate average acc from the start of current epoch till current batch
        acc_current_average = topkacc.accumulate()
    # calculate acc of the epoch
    acc_epoch = topkacc.accumulate()

>>> acc_epoch
{'top_1': 0.6, 'top_2': 0.8, 'top_5': 0.9}

```

## Contact

This repository is currently maintained by Chenhao Wang ([@C-H-Wong](http://github.com/C-H-Wong)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got and modified much code from [scikit-learn](https://github.com/scikit-learn/scikit-learn), thanks to [scikit-learn](https://github.com/scikit-learn/scikit-learn).
