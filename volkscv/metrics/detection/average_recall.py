from .base import COCOAnalysis


class AverageRecall(COCOAnalysis):

    def compute(self, pred, target):
        super().compute(pred, target)
        state = self._display_set()
        return state

    def _display_set(self):

        mar_stats = []
        ar_stats = []

        if (self.iou is None) and (self.areaRng is None):
            # COCO map setting
            for dets in self.cocoEval.params.maxDets:
                ar_stats.append(
                    self._summarize(0, maxDets=dets))
            ar_stats.append(
                self._summarize(0, areaRng='small', maxDets=self.cocoEval.params.maxDets[-1]))
            ar_stats.append(
                self._summarize(0, areaRng='medium', maxDets=self.cocoEval.params.maxDets[-1]))
            ar_stats.append(
                self._summarize(0, areaRng='large', maxDets=self.cocoEval.params.maxDets[-1]))

        else:
            if self.iou is None:
                ar_iou = [0.5, 0.75]
            else:
                ar_iou = self.cocoEval.params.iouThrs

            if self.areaRngLbl is None:
                self.areaRngLbl = ['all', 'small', 'medium', 'large']

            if len(ar_iou) != 1:
                for label in self.areaRngLbl:
                    for dets in self.cocoEval.params.maxDets:
                        mar_stats.append(
                            self._summarize(0, areaRng=label, maxDets=dets))

            for iou in ar_iou:
                for label in self.areaRngLbl:
                    for dets in self.cocoEval.params.maxDets:
                        ar_stats.append(
                            self._summarize(0, iouThr=iou, areaRng=label, maxDets=dets))

        accumulate_state = {
            'mar': mar_stats,
            'ar': ar_stats
        }
        return accumulate_state
