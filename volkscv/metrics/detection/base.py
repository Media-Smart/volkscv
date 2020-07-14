from abc import ABCMeta, abstractmethod

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
    def __init__(self):
        super().__init__()

    def reset(self):
        self.cocoEval = None
        self.precision = None
        self.recall = None
        self.score = None

    def compute(self, pred_path, target_path):
        coco = COCO(target_path)
        cocoDT = coco.loadRes(pred_path)
        self.cocoEval = COCOeval(coco, cocoDT, 'bbox')

    def accumulate(self):
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.cocoEval.summarize()
        self.precision = self.cocoEval.eval['precision']
        self.recall = self.cocoEval.eval['recall']
        self.score = self.cocoEval.eval['scores']
        return None
