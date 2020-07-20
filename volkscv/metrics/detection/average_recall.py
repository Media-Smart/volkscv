import numpy as np

from .base import COCOAnalysis


class AverageRecall(COCOAnalysis):

    def __init__(self, ar_iou=None):
        super().__init__()
        self.ar_iou = ar_iou

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        if self.ar_iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ar_iou)), axis=0)

    def accumulate(self):
        super().accumulate()

        ar_stats = []
        if self.ar_iou is None:
            # COCO ar setting
            ar_stats.append(
                self._summarize(0, maxDets=self.cocoEval.params.maxDets[0]))
            ar_stats.append(
                self._summarize(0, maxDets=self.cocoEval.params.maxDets[1]))
            ar_stats.append(
                self._summarize(0, maxDets=self.cocoEval.params.maxDets[2]))
            ar_stats.append(
                self._summarize(0, areaRng='small', maxDets=self.cocoEval.params.maxDets[2]))
            ar_stats.append(
                self._summarize(0, areaRng='medium', maxDets=self.cocoEval.params.maxDets[2]))
            ar_stats.append(
                self._summarize(0, areaRng='large', maxDets=self.cocoEval.params.maxDets[2]))
        else:
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
        print(accumulate_state)
        return accumulate_state
