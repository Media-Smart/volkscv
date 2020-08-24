from abc import ABCMeta, abstractmethod

import numpy as np


class BaseMetric(object, metaclass=ABCMeta):
    """
    Base metric for metrics.
    This class is abstract, providing a standard interface for metrics.
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
    def compute(self, pred, target):
        """
        Compute metric value for current batch for iterable kind of metrics, eg, Accuracy, etc,
        or compute process value for statistic kind of metrics, eg, PR_Cure, etc.

        Args:
            pred (numpy.ndarray): prediction results from classification model,
                pred should have the following shape (batch_size, num_categories),
                pred should stand for the predicted probability with sum==1
            target (numpy.ndarray): ground truth  class indices,
                target should have the following shape (batch_size)
        Returns:
            metric value or process value for current batch
        """
        pass

    def update(self, n=1):
        """
        Add metric value or process value to statistic containers for on-line metrics.
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
        self._check_match(pred, target)

    @staticmethod
    def _check_match(pred, target):
        assert pred.shape[0] == target.shape[0], \
            "pred and target don't match"

    @staticmethod
    def _check_type(pred, target):
        assert type(pred) == np.ndarray and type(target) == np.ndarray, \
            "Only numpy.ndarray is supported for computing accuracy"

    @staticmethod
    def _check_pred_range(pred):
        assert np.all(0 <= pred) and np.all(pred <= 1), \
            "Pred should stand for the predicted probability in range (0, 1)"

    def __call__(self, pred, target):
        self.check(pred, target)
        current_state = self.compute(pred, target)
        self.update()
        return current_state
