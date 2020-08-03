import numpy as np

from .base import COCOAnalysis


class AveragePrecision(COCOAnalysis):

    def __init__(self, ap_iou=None, maxdets=None, areaRng=None, areaRngLbl=None):
        super().__init__()
        self.ap_iou = ap_iou
        self.maxdets = maxdets
        self.areaRng = areaRng
        self.areaRngLbl = areaRngLbl

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        if self.ap_iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ap_iou)), axis=0)
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

        ap_stats = []
        ap_stats.append(
            self._summarize(1, maxDets=self.cocoEval.params.maxDets[-1]))

        if (self.ap_iou is None) and (self.areaRng is None):
            # COCO map setting
            ap_stats.append(
                self._summarize(1, iouThr=.5, maxDets=self.cocoEval.params.maxDets[-1]))
            ap_stats.append(
                self._summarize(1, iouThr=.75, maxDets=self.cocoEval.params.maxDets[-1]))
            ap_stats.append(
                self._summarize(1, areaRng='small', maxDets=self.cocoEval.params.maxDets[-1]))
            ap_stats.append(
                self._summarize(1, areaRng='medium', maxDets=self.cocoEval.params.maxDets[-1]))
            ap_stats.append(
                self._summarize(1, areaRng='large', maxDets=self.cocoEval.params.maxDets[-1]))

        else:
            if self.ap_iou is None:
                self.ap_iou = [0.5, 0.75]
            if self.areaRngLbl is None:
                self.areaRngLbl = ['all', 'small', 'medium', 'large']

            for iou in self.cocoEval.params.iouThrs:
                ap_stats.append(
                    self._summarize(1, iouThr=iou, maxDets=self.cocoEval.params.maxDets[-1]))
            for label in self.areaRngLbl:
                ap_stats.append(
                    self._summarize(1, areaRng=label, maxDets=self.cocoEval.params.maxDets[-1]))
            for iou in self.cocoEval.params.iouThrs:
                for label in self.areaRngLbl:
                    ap_stats.append(
                        self._summarize(1, iouThr=iou, areaRng=label, maxDets=self.cocoEval.params.maxDets[-1]))

        accumulate_state = {
            'map': ap_stats[0],
            'ap': ap_stats[1:]
        }
        return accumulate_state
