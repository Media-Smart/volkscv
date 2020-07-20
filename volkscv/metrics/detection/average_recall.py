import numpy as np

from .base import COCOAnalysis


class AverageRecall(COCOAnalysis):

    def __init__(self, ar_iou=None, maxdets=None):
        super().__init__()
        self.ar_iou = ar_iou
        self.maxdets = maxdets

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        if self.ar_iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ar_iou)), axis=0)
        if self.maxdets is not None:
            assert type(self.maxdets) is list, 'maxdets must be a list'
            self.cocoEval.params.maxDets = list(set(self.maxdets))
            self.cocoEval.params.maxDets.sort()

    def accumulate(self):
        super().accumulate()

        ar_stats = []
        for dets in self.cocoEval.params.maxDets:
            ar_stats.append(
                self._summarize(0, maxDets=dets))
        for iou in self.cocoEval.params.iouThrs:
            for dets in self.cocoEval.params.maxDets:
                ar_stats.append(
                    self._summarize(0, iouThr=iou, maxDets=dets))

        index = len(self.cocoEval.params.maxDets)
        accumulate_state = {
            'mar': ar_stats[0:index],
            'ar': ar_stats[index:]
        }
        return accumulate_state
