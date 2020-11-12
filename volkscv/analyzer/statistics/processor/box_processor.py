import matplotlib.pyplot as plt
import numpy as np

from .base import BaseProcessor
from ..plotter import OneDimPlotter, TwoDimPlotter, SubPlotter, Compose, cdf_pdf


class BoxProcessor(BaseProcessor):
    """ Process the information related to box, get several statistical distribution.

    Args:
        data (dict): Data to be processed.

    Example:
        >>> import numpy as np
        >>> data = dict(
        >>>     bboxes=np.array([np.array([[0, 0, 10, 10], [15, 30, 40, 60]]),
        >>>                      np.array([[10, 15, 20, 20]]),]),
        >>>     labels = np.array([np.array([0, 1]), np.array([1])]),
        >>>     )
        >>> self = BoxProcessor(data)
        >>> self.default_plot()
        >>> # export
        >>> self.export('./result', save_mode='folder')
    """

    def __init__(self, data):
        super(BoxProcessor, self).__init__(data)
        self.processor = ['hw', 'scale', 'ratio', 'ratio_log2', 'hw_per_class']
        self._sections = [[0, 1e8]]
        self._box_h, self._box_w = self._extract_box()
        self._box_per_class = self._box_of_each_class()
        self.box_per_class = None
        self.box_h, self.box_w = None, None
        self._text = 'all'
        if not self._box_h:
            self.processor = []

    def _extract_box(self):
        """ Extract the box height and width in input data."""

        box_h = []
        box_w = []
        if self.data.get('bboxes', None) is not None:
            for boxs in self.data['bboxes']:
                for box in boxs:
                    h, w = box[2:] - box[:2]
                    box_h.append(h)
                    box_w.append(w)
        else:
            print("Keys in data doesn't contain 'labels'.")

        return box_h, box_w

    def _box_of_each_class(self):
        """ Divide the height and width of box into different groups based on
         their class.

         Returns:
             _box_per_class (dict): dict(category: [[h1, h2...], [w1, w2...]])
         """

        if self.data.get('labels', None) is None:
            print("Keys in data doesn't contain 'labels'.")
            return None
        if not self._box_h:
            return None
        label = self.data['labels']
        tmp_label = []
        for l in label:
            tmp_label += list(l)
        if 'categories' in self.data and self.data['categories'] is not None:
            categories = list(range(len(self.data['categories'])))
        else:
            categories = list(set(tmp_label))
        self._class = categories
        box_per_class = {categories[tl]: [[], []] for tl in set(tmp_label)}
        for cl, ch, cw in zip(tmp_label, self._box_h, self._box_w):
            box_per_class[categories[cl]][0].append(ch)
            box_per_class[categories[cl]][1].append(cw)

        return box_per_class

    @property
    def specified_class(self):
        return self._class

    @specified_class.setter
    def specified_class(self, v):
        if not isinstance(v, (list, tuple)):
            v = [v]
        for v_ in v:
            assert isinstance(v_, int), "Use int value to specify class."
        self._class = v
        h, w = [], []
        for sc in self.specified_class:
            h += self._box_per_class[sc][0]
            w += self._box_per_class[sc][1]
        self.box_per_class = {sc: self._box_per_class[sc] for sc in v}
        self.box_h, self.box_w = h, w
        self._text = str(v)

    @property
    def sections(self):
        """ The section of box scale (sqrt(box_w*box_h))."""
        return self._sections

    @sections.setter
    def sections(self, v):
        assert isinstance(v, (list, tuple))
        assert isinstance(v[0], (int, float))
        v = [0] + v + [1e8]
        self._sections = [[v[idx], v[idx + 1]] for idx in range(len(v) - 1)]

    @property
    def hw_per_class(self):
        """Height and width distribution of each class."""

        if self._box_per_class is None:
            return None

        if self.box_per_class is not None:
            unique_class = self.box_per_class
        else:
            unique_class = self._box_per_class
        cols = int(np.ceil(np.sqrt(len(unique_class))))
        return SubPlotter(unique_class,
                          'box hw distribution of class %s' % self._text,
                          'two',
                          plt.scatter,
                          cols, cols,
                          axis_label=['height', 'width'],
                          marker='.',
                          alpha=0.1)

    @property
    def hw(self):
        """ Height and width distribution of box. """
        h, w = self._box_h, self._box_w
        if self.box_h:
            h, w = self.box_h, self.box_w

        return TwoDimPlotter([h, w],
                             "distribution of box's hw (class %s)" % self._text,
                             plt.scatter,
                             axis_label=['height', 'width'],
                             marker='.', alpha=0.1)

    @property
    def scale(self):
        """ Scale (sqrt(w*h)) distribution."""
        h, w = self._box_h, self._box_w
        if self.box_h:
            h, w = self.box_h, self.box_w

        sqrt_scale = np.sqrt(np.array(w) * np.array(h))

        return OneDimPlotter(list(sqrt_scale), 'sqrt(wh) of box (class %s)' % self._text,
                             cdf_pdf, axis_label=['scale:sqrt(wh)', 'normalized numbers'],
                             bins=20)

    def section_scale(self, srange=(0, 32, 96, 640)):
        """ Scale (sqrt(w*H)) distribution in different sections."""
        # TODO
        sections = [[srange[idx], srange[idx + 1]] for idx in range(len(srange) - 1)]
        print('The sections are %s' % sections)
        sqrt_scale = np.sqrt(np.array(self._box_w) * np.array(self._box_h))
        return OneDimPlotter(sqrt_scale, 'box nums in different section' % sections,
                             cdf_pdf, axis_label=['scale:sqrt(wh)', 'normalized numbers'],
                             bins=srange)

    @property
    def ratio(self):
        """ Ratio (height/width) distribution."""

        assert min(self._box_w) > 0
        h, w = self._box_h, self._box_w
        if self.box_h:
            h, w = self.box_h, self.box_w
        section_hw = {i: [[], []] for i in range(len(self.sections))}
        for h_, w_ in zip(h, w):
            for idx, section in enumerate(self.sections):
                if section[0] <= np.sqrt(h_ * w_) < section[1]:
                    section_hw[idx][0].append(h_)
                    section_hw[idx][1].append(w_)
        legends = []
        plotters = []
        for key, value in section_hw.items():
            hw_ratios = np.array(value[0]) / np.array(value[1])
            legends.append(self.sections[key])
            plotters.append(OneDimPlotter(list(hw_ratios),
                                          'h w ratio of box (class %s) in section %s' %
                                          (self.sections[key], self._text),
                                          cdf_pdf,
                                          axis_label=['h/w ratio', 'normalized numbers'],
                                          bins=20))
        return Compose(plotters, text='Box ratio of class %s' % self._text, legend=legends)

    @property
    def ratio_log2(self):
        """ Ratio (log2(height/width)) distribution."""

        assert min(self._box_w) > 0
        h, w = self._box_h, self._box_w
        if self.box_h:
            h, w = self.box_h, self.box_w

        h_w_ratio = np.array(h) / np.array(w)
        log2_ratio = np.log2(h_w_ratio)
        return OneDimPlotter(list(log2_ratio),
                             'h/w ratio(log2) of box (class %s)' % self._text,
                             cdf_pdf,
                             axis_label=['log2(h/w)', 'normalized numbers'],
                             bins=20)
