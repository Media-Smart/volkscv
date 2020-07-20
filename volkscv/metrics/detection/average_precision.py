import numpy as np

from .base import COCOAnalysis


class AveragePrecision(COCOAnalysis):

    def __init__(self, ap_iou=None, maxdets=None):
        super().__init__()
        self.ap_iou = ap_iou
        self.maxdets = maxdets

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        # self.ap_iou = np.linspace(0.1, 0.95, np.round((0.95 - .1) / .05) + 1, endpoint=True).round(3)
        if self.ap_iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ap_iou)), axis=0)
        if self.maxdets is not None:
            assert type(self.maxdets) is list, 'maxdets must be a list'
            self.cocoEval.params.maxDets = self.maxdets

    def accumulate(self):
        super().accumulate()

        ap_stats = []
        if self.ap_iou is None:
            # COCO map setting
            ap_stats.append(
                self._summarize(1, maxDets=max(self.cocoEval.params.maxDets)))
            ap_stats.append(
                self._summarize(1, iouThr=.5, maxDets=max(self.cocoEval.params.maxDets)))
            ap_stats.append(
                self._summarize(1, iouThr=.75, maxDets=max(self.cocoEval.params.maxDets)))
        else:
            ap_stats.append(
                self._summarize(1, maxDets=max(self.cocoEval.params.maxDets)))
            for iou in self.cocoEval.params.iouThrs:
                ap_stats.append(
                    self._summarize(1, iouThr=iou, maxDets=max(self.cocoEval.params.maxDets)))

        accumulate_state = {
            'map': ap_stats[0],
            'ap': ap_stats[1:]
        }
        return accumulate_state
