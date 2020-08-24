from abc import ABCMeta, abstractmethod

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class BaseMetric(object, metaclass=ABCMeta):
    """
    Base metric for detection metrics.
    This class is abstract, providing a standard interface for metrics of this type.
    """
    def __init__(self):
        super(BaseMetric, self).__init__()
        self.reset()

    @abstractmethod
    def reset(self):
        """
        Reset variables to default settings.
        """
        pass

    @abstractmethod
    def compute(self, pred_path, target_path):
        """
        Compute metric value for current batch or compute process value for metrics.

        Args:
            pred_path (str): path to results file, prediction results from detection model,
                stored in a dict, following the format of COCO, saved in a json file.
                [{'image_id': XX, 'bbox': [x, y, w, h], 'score': X, 'category_id': X }, ...]
            target_path (str): path to ground truth file following the format of COCO
                annotation, saved in a json file.
        Returns:
            metric value or process value for current batch
        """
        pass

    @abstractmethod
    def accumulate(self):
        """
        Compute accumulated metric value.
        """
        pass

    def export(self):
        """
        Export figures, images or reports of metrics
        """
        pass

    def check(self, pred, target):
        """
        Check inputs
        """
        self._check_type(pred, target)

    @staticmethod
    def _check_type(pred, target):
        assert type(pred) == str and type(target) == str, \
            "Inputs refers to the path to json file"

    def __call__(self, pred, target):
        self.check(pred, target)
        self.compute(pred, target)
        accumulate_stat = self.accumulate()
        return accumulate_stat


class COCOAnalysis(BaseMetric):
    """
    Basic analysis using pycocotools.
    """
    def __init__(self, iou=None, maxdets=None, areaRng=None, areaRngLbl=None):
        super().__init__()
        self.iou = iou
        self.maxdets = maxdets
        self.areaRng = areaRng
        self.areaRngLbl = areaRngLbl

    def reset(self):
        self.cocoEval = None
        self.precision = None
        self.recall = None
        self.score = None

    def compute(self, pred_path, target_path):
        coco = COCO(target_path)
        cocoDT = coco.loadRes(pred_path)
        self.cocoEval = COCOeval(coco, cocoDT, 'bbox')

        if self.iou is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.iou)), axis=0)
        if self.maxdets is not None:
            assert type(self.maxdets) is list, 'maxdets must be a list'
            self.cocoEval.params.maxDets = list(set(self.maxdets))
            self.cocoEval.params.maxDets.sort()
        if self.areaRng is not None:
            assert self.areaRngLbl is not None, 'areaRngLbl must be specified for self-defined areaRng!'
            assert len(self.areaRng) == len(self.areaRngLbl), 'areaRng and areaRngLbl must be match!'
            self.cocoEval.params.areaRng = self.areaRng
            self.cocoEval.params.areaRngLbl = self.areaRngLbl

    def accumulate(self):
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.precision = self.cocoEval.eval['precision']
        self.recall = self.cocoEval.eval['recall']
        self.score = self.cocoEval.eval['scores']

    def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.cocoEval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.precision
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.recall
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
