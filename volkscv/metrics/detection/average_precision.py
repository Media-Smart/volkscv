from .base import COCOAnalysis


class AveragePrecision(COCOAnalysis):

    def compute(self, pred, target):
        super().compute(pred, target)
        state = self._display_set()
        return state

    def _display_set(self):

        map_stats = []
        ap_stats = []

        if (self.iou is None) and (self.areaRng is None):
            # COCO map setting
            ap_stats.append(
                self._summarize(1, maxDets=self.cocoEval.params.maxDets[-1]))
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
            if self.iou is None:
                ap_iou = [0.5, 0.75]
            else:
                ap_iou = self.cocoEval.params.iouThrs

            if self.areaRngLbl is None:
                self.areaRngLbl = ['all', 'small', 'medium', 'large']

            if len(ap_iou) != 1:
                for label in self.areaRngLbl:
                    map_stats.append(
                        self._summarize(1, areaRng=label, maxDets=self.cocoEval.params.maxDets[-1]))
            for iou in ap_iou:
                for label in self.areaRngLbl:
                    ap_stats.append(
                        self._summarize(1, iouThr=iou, areaRng=label, maxDets=self.cocoEval.params.maxDets[-1]))

        accumulate_state = {
            'map': map_stats,
            'ap': ap_stats
        }
        return accumulate_state
