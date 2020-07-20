import numpy as np

from .base import COCOAnalysis


class AveragePrecision(COCOAnalysis):

    def __init__(self, ap_iou=None):
        super().__init__()
        self.ap_iou = ap_iou

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        if self.ap_iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ap_iou)), axis=0)

    def accumulate(self):
        super().accumulate()

        ap_stats = []
        if self.ap_iou is None:
            # COCO ap setting
            ap_stats.append(
                self._summarize(1))
            ap_stats.append(
                self._summarize(1, iouThr=.5, maxDets=self.cocoEval.params.maxDets[2]))
            ap_stats.append(
                self._summarize(1, iouThr=.75, maxDets=self.cocoEval.params.maxDets[2]))
            ap_stats.append(
                self._summarize(1, areaRng='small', maxDets=self.cocoEval.params.maxDets[2]))
            ap_stats.append(
                self._summarize(1, areaRng='medium', maxDets=self.cocoEval.params.maxDets[2]))
            ap_stats.append(
                self._summarize(1, areaRng='large', maxDets=self.cocoEval.params.maxDets[2]))
        else:
            ap_stats.append(self._summarize(1))
            for iou in self.cocoEval.params.iouThrs:
                ap_stats.append(
                    self._summarize(1, iouThr=iou, maxDets=self.cocoEval.params.maxDets[2]))

        accumulate_state = {
            'map': ap_stats[0],
            'ap': ap_stats[1:]
        }
        return accumulate_state
