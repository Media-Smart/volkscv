# modify from sklearn (0.22.1)
from functools import partial

import numpy as np

from .precision_recall_curve import precision_recall_curve
from .utils.multiclass import type_of_target
from .utils._base import _average_binary_score


def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
    """Compute average precision (AP) from prediction scores

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    pos_label : int or str (default=1)
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    average_precision : float

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)
    0.83...

    Notes
    -----
    .. versionchanged:: 0.19
      Instead of linearly interpolating between operating points, precisions
      are weighted by the change in recall since the last operating point.
    """
    def _binary_uninterpolated_average_precision(
            y_true, y_score, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        present_labels = np.unique(y_true)
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError("pos_label=%r is invalid. Set it to a label in "
                             "y_true." % pos_label)
    average_precision = partial(_binary_uninterpolated_average_precision,
                                pos_label=pos_label)
    return _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
