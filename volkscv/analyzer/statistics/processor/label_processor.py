from .base import BaseProcessor
from ..plotter import OneDimPlotter, cdf_pdf


class LabelProcessor(BaseProcessor):
    """ Process the information related to labels, get several statistical distribution.

    Args:
        data (dict): Data to be processed.

    Examples:
        >>> import numpy as np
        >>> data = dict(
        >>>     labels = np.array([np.array([0, 1]), np.array([1])]),
        >>>     )
        >>> self = LabelProcessor(data)
        >>> self.default_plot()
        >>> # export
        >>> self.export('./result', save_mode='folder')
        >>> # what statistical data processed
        >>> print(self.processor)
    """

    def __init__(self, data):
        super(LabelProcessor, self).__init__(data)
        self.processor = ['labels']
        if self.data.get('labels', None) is None:
            self.processor = []

    @property
    def labels(self):
        """Distribution of each class's number."""

        if self.data.get('labels', None) is None:
            return None

        label = self.data['labels']
        tmp = []
        for l in label:
            tmp += l.flatten().tolist()
        label = tmp
        bins_ = len(set(label))
        return OneDimPlotter(label, 'label distribution', cdf_pdf,
                             axis_label=['label', 'normalized number'],
                             bins=bins_)
