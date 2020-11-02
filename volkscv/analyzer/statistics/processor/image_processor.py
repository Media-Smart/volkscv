import matplotlib.pyplot as plt
import numpy as np

from .base import BaseProcessor
from ..plotter import OneDimPlotter, TwoDimPlotter, cdf_pdf


class ImageProcessor(BaseProcessor):
    """ Process the information related to image, get several statistical distribution.

    Args:
        data (dict): Data to be processed.

    Examples:
        >>> import numpy as np
        >>> data = dict(
        >>>     shapes=np.array([np.array([100,300]), np.array([150, 1000])]),
        >>>     labels = np.array([np.array([0, 1]), np.array([1])]),
        >>>     )
        >>> self = ImageProcessor(data)
        >>> self.default_plot()
        >>> # export
        >>> self.export('./result', save_mode='folder')
        >>> # what statistical data processed
        >>> print(self.processor)
    """

    def __init__(self, data):
        super(ImageProcessor, self).__init__(data)
        self.processor = ['hw', 'ratio', 'scale', 'ratio_log2', 'instances_per_image']
        if self.data.get('shapes', None) is None:
            print("Image size distribution, ratio distribution, scale distribution"
                  " and log2(ratio) is related to 'shapes'. "
                  "But got no 'shapes' in input data.")
            self.processor = ['instances_per_image']
        if self.data.get('labels', None) is None:
            print("Instances per image is related to 'labels'. "
                  "But got no 'labels' in input data.")
            self.processor.remove('instances_per_image')

    @property
    def hw(self):
        """Height and width distribution of image."""

        if self.data.get('shapes', None) is None:
            return None
        h, w = self.data['shapes'][:, 0], self.data['shapes'][:, 1]
        return TwoDimPlotter([h, w], 'image hw distribution', plt.scatter,
                             axis_label=['height', 'width'],
                             marker='.', alpha=0.1)

    @property
    def ratio(self):
        """Ratio (height/width) distribution of image."""

        if self.data.get('shapes', None) is None:
            return None
        hw_ratio = self.data['shapes'][:, 0] / self.data['shapes'][:, 1]

        return OneDimPlotter(hw_ratio, r'image h/w ratio',
                             cdf_pdf,
                             axis_label=['ratio: h/w', 'normalized number'],
                             bins=20)

    @property
    def ratio_log2(self):
        """Ratio (log2(height/width)) distribution of image."""

        if self.data.get('shapes', None) is None:
            return None
        hw_ratio = self.data['shapes'][:, 0] / self.data['shapes'][:, 1]
        log_ratio = np.log2(hw_ratio)

        return OneDimPlotter(log_ratio, r'image h/w ratio (log2)',
                             cdf_pdf,
                             axis_label=['ratio: log2(h/2)', 'normalized number'],
                             bins=20)

    @property
    def scale(self):
        """Scale (sqrt(width*height)) distribution of image."""

        if self.data.get('shapes', None) is None:
            return None
        h, w = self.data['shapes'][:, 0], self.data['shapes'][:, 1]
        sqrt_hw = np.sqrt(h * w)
        range_ = (np.min(sqrt_hw), np.max(sqrt_hw))

        return OneDimPlotter(sqrt_hw, r'image Scale(diagonal length)',
                             cdf_pdf,
                             axis_label=['scale: sqrt(wh)', 'normalized number'],
                             bins=20, range=range_)

    @property
    def instances_per_image(self):
        """Distribution of instance numbers per image."""

        if self.data.get('labels', None) is None:
            return None

        label = self.data['labels']
        label_ = [l.size for l in label]

        return OneDimPlotter(label_, 'instance nums per image', plt.hist,
                             axis_label=['instance nums per image', 'normalized number'], )
