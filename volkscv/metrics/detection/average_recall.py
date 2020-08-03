import numpy as np

from .base import COCOAnalysis


class AverageRecall(COCOAnalysis):

    def __init__(self, ar_iou=None, maxdets=None, areaRng=None, areaRngLbl=None):
        super().__init__()
        self.ar_iou = ar_iou
        self.maxdets = maxdets
        self.areaRng = areaRng
        self.areaRngLbl = areaRngLbl

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        if self.ar_iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ar_iou)), axis=0)
        if self.maxdets is not None:
            assert type(self.maxdets) is list, 'maxdets must be a list'
            self.cocoEval.params.maxDets = list(set(self.maxdets))
            self.cocoEval.params.maxDets.sort()
        if self.areaRng is not None:
            assert self.areaRngLbl is not None, 'areaRng and areaRngLbl must be match!'
            assert len(self.areaRng) == len(self.areaRngLbl), 'areaRng and areaRngLbl must be match!'
            self.cocoEval.params.areaRng = self.areaRng
            self.cocoEval.params.areaRngLbl = self.areaRngLbl

    def accumulate(self):
        super().accumulate()

        ar_stats = []
        for dets in self.cocoEval.params.maxDets:
            ar_stats.append(
                self._summarize(0, maxDets=dets))

        if (self.ar_iou is None) and (self.areaRng is None):
            ar_stats.append(
                self._summarize(0, areaRng='small', maxDets=self.cocoEval.params.maxDets[-1]))
            ar_stats.append(
                self._summarize(0, areaRng='medium', maxDets=self.cocoEval.params.maxDets[-1]))
            ar_stats.append(
                self._summarize(0, areaRng='large', maxDets=self.cocoEval.params.maxDets[-1]))

        else:
            if self.ar_iou is None:
                ar_iou = [0.5, 0.75]
            else:
                ar_iou = self.cocoEval.params.iouThrs

            if self.areaRngLbl is None:
                self.areaRngLbl = ['all', 'small', 'medium', 'large']

            for iou in ar_iou:
                for label in self.areaRngLbl:
                    for dets in self.cocoEval.params.maxDets:
                        ar_stats.append(
                            self._summarize(0, iouThr=iou, areaRng=label, maxDets=dets))

        index = len(self.cocoEval.params.maxDets)
        accumulate_state = {
            'mar': ar_stats[0:index],
            'ar': ar_stats[index:]
        }
        return accumulate_state
