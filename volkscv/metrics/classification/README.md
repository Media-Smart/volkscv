## Support
- [x] [TopKAccuracy](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/accuracy.py)
- [x] [ConfusionMatrix](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/confusion_matrix.py)
- [x] [CMAccuracy](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/confusion_matrix.py)
- [x] [CMPrecisionRecall](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/confusion_matrix.py)
- [x] [Fbetascore](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/fbeta_score.py)
- [x] [PRCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/pr_curve.py)
- [x] [ROCCurve](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/roc_curve.py)
- [x] [APscore](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/average_precision_score.py)
- [x] [mAPscore](https://github.com/Media-Smart/volkscv/blob/master/volkscv/metrics/classification/average_precision_score.py)

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
{'top_1': 0.6, 'top_3': 0.8, 'top_5': 0.9}

```
