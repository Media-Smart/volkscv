import numpy as np
import torch

from volkscv.utils.geos import bbox_overlaps
from .base import BaseProcessor
from ..plotter import cdf_pdf, OneDimPlotter, Compose


class AnchorProcessor(BaseProcessor):
    """ Process the data, get several statistical distribution.

    Args:
        data (dict): Parsed data using ``volkscv.parser.parser``.
        anchor_generator (volkscv.analysis.BaseAnchorGenerator): Anchor generator.
        target_shape (tuple): Image shape after resize. Current resize method is mmdetection version.
        assigner (volkscv.analysis.BaseAssigner): Assigner is used to assign anchors to different gt bboxes.

    Example:
        >>> import numpy as np
        >>> from volkscv.analyzer.statistics.utils import AnchorGenerator, MaxIoUAssigner
        >>> data = dict(img_names=['./img/a.png'], labels=[[1]],
        >>>             shapes=[[800, 1333]], categories=['cat'],
        >>>             bboxes=[[[0, 0, 10, 10]]], segs=[[[0,0,1,1,2,2,3,3]]])
        >>> self = AnchorProcessor(data, (800, 1333), AnchorGenerator, MaxIoUAssigner)
        >>> self.default_plot()
        >>> self.show()
        >>> self.export('./result', save_mode='folder')
        >>> self.clear()
    """

    def __init__(self, data, target_shape, anchor_generator, assigner=None, device='cuda'):
        super(AnchorProcessor, self).__init__(data)
        self.anchor_generator = anchor_generator
        self.img_size = data['shapes'][0]
        self.assigner = assigner
        self._range_iou_distribution = None
        self._areas = None
        self._max_ious = None
        self._unique_ratio = None
        self._anchor_coverage_nums = None
        self._range_iou_distribution = None
        self._range_anchor_coverage = None
        self._max_assigner_iou = None
        self._sections = [[0, 1000000]]
        self._process_anchor(target_size=target_shape, device=device)
        self.processor = ['max_iou_distribution', 'max_assigner_iou_distribution',
                          'section_iou_distribution', 'anchor_coverage', 'section_anchor_coverage']

    def _get_feature_map_size(self, img_size):
        if self.anchor_generator is not None:
            strides = self.anchor_generator.strides
            featmap_sizes = [(np.array(img_size) / stride[0]).astype('int') for stride in strides]
            return featmap_sizes

        return None

    def _generate_index(self, bboxes, area_range):

        arae_dict = dict()
        for idx, a_r in enumerate(area_range):
            for idx2, box in enumerate(bboxes):
                t_area = (box[0] - box[2]) * (box[1] - box[3])
                if a_r[0] ** 2 <= t_area < a_r[1] ** 2:
                    if str(idx) not in arae_dict:
                        arae_dict[str(idx)] = [idx2]
                    else:
                        arae_dict[str(idx)].append(idx2)
        return arae_dict

    @staticmethod
    def _generate_area(bboxes):
        areas = []
        for idx2, box in enumerate(bboxes):
            t_area = (box[0] - box[2]) * (box[1] - box[3])
            areas.append(t_area)
        return areas

    def _resize(self, shape, target_shape, bboxes, ignore_bboxes=None):
        w, h = shape
        max_long_edge = max(target_shape)
        max_short_edge = min(target_shape)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))

        shape = (int(h * scale_factor + 0.5), int(w * scale_factor + 0.5))
        w_s, h_s = shape[1] / w, shape[0] / h
        scale_factor = np.array([w_s, h_s, w_s, h_s], dtype=np.float32)
        bboxes = (bboxes * scale_factor).astype(np.float32)
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, shape[0])
        if ignore_bboxes is not None:
            ignore_bboxes = (ignore_bboxes * scale_factor).astype(np.float32)
            ignore_bboxes[:, 0::2] = np.clip(ignore_bboxes[:, 0::2], 0, shape[1])
            ignore_bboxes[:, 1::2] = np.clip(ignore_bboxes[:, 1::2], 0, shape[0])

        return shape, bboxes, ignore_bboxes

    def _process_anchor(self, target_size=None, device='cuda'):
        max_ious = []
        nums_list = []
        max_assigner_ious = []
        unique_nums = 0
        areas = []
        for bboxes, ignore_bboxes, labels, shape in zip(self.data['bboxes'], self.data['bboxes_ignore'],
                                                        self.data['labels'], self.data['shapes']):
            areas.append(self._generate_area(bboxes))
            if target_size is not None:
                shape, bboxes, ignore_bboxes = self._resize(shape, target_size, bboxes, ignore_bboxes)
            else:
                bboxes = bboxes.astype(np.float32)
                ignore_bboxes = ignore_bboxes.astype(np.float32)
                shape = shape[::-1]

            if bboxes.shape[0] == 0:
                continue
            feat_size = self._get_feature_map_size(shape)
            multi_level_anchors = self.anchor_generator.grid_anchors(feat_size, device)

            gt_bboxes = torch.from_numpy(bboxes).to(device)
            gt_bboxes_ignore = torch.from_numpy(ignore_bboxes).to(device)
            anchors = torch.cat(multi_level_anchors, 0).to(device)

            # no assigner
            iou_mat = bbox_overlaps(gt_bboxes, anchors)
            max_iou, max_index = torch.max(iou_mat, 1)
            max_ious.append(max_iou.cpu().detach().numpy())
            unique_nums += len(set(max_index.cpu().detach().numpy()))

            # assigner
            if self.assigner is not None:
                gt_inds, gt_ious = self.assigner.assign(anchors,
                                                        gt_bboxes,
                                                        gt_bboxes_ignore)
                max_assigner_iou = []
                for i in range(1, len(bboxes) + 1):
                    iou = gt_ious[gt_inds == i]
                    if iou.size(0) != 0:
                        max_assigner_iou.append(torch.max(iou).detach().cpu().numpy())
                    else:
                        max_assigner_iou.append(-1)

                nums = np.bincount(gt_inds)[1:]
                nums_list.append(nums)
                max_assigner_ious.append(max_assigner_iou)

        self.max_iou_distribution = np.concatenate(max_ious, 0)
        if max_assigner_ious:
            self.max_assigner_iou_distribution = np.concatenate(max_assigner_ious, 0)
        if nums_list:
            self.anchor_coverage = np.concatenate(nums_list, axis=0)
        self.areas = np.concatenate(areas, 0)
        self.unique_ratio = unique_nums / len(self.max_iou_distribution.data)
        print(f'max iou matching unique ratio is {self.unique_ratio}')

    def specify_feat_map(self, taregt_size, base_anchors, strides, device='cuda'):
        # TODO
        self.anchor_generator.base_anchors = base_anchors
        self.anchor_generator.strides = strides
        self._process_anchor(taregt_size, device)

    @property
    def unique_ratio(self):
        """ Compute the iou matrix, assign the anchor index to the gt if the iou of
        the anchor and gt is the maximum. Then, unique_ratio=len(set(indexes))/len(indexs).
        """

        return self._unique_ratio

    @unique_ratio.setter
    def unique_ratio(self, v):
        self._unique_ratio = v

    @property
    def max_iou_distribution(self):
        """ Distrubution of each gt bbox's max IoU."""

        return self._max_ious

    @max_iou_distribution.setter
    def max_iou_distribution(self, v):
        self._max_ious = OneDimPlotter(v, 'max iou distribution', cdf_pdf,
                                       axis_label=['iou', 'normalized numbers'],
                                       bins=10, range=(0, 1))

    @property
    def max_assigner_iou_distribution(self):
        """Distribution of each gt bbox's max IoU after assigner."""

        if self._max_assigner_iou is None:
            print(f"Assigner shouldn't be None.")
        return self._max_assigner_iou

    @max_assigner_iou_distribution.setter
    def max_assigner_iou_distribution(self, v):
        self._max_assigner_iou = OneDimPlotter(v, 'max assigner iou distribution', cdf_pdf,
                                               axis_label=['iou', 'normalized numbers'],
                                               bins=10, range=(0, 1))

    @property
    def areas(self):
        """List of box area."""
        return self._areas

    @areas.setter
    def areas(self, v):
        self._areas = v

    @property
    def sections(self):
        """ Partition the box based on the sections."""
        return self._sections

    @sections.setter
    def sections(self, v):
        self._sections = v

    @property
    def section_iou_distribution(self):
        """Distribution of max IoU in different sections."""
        data_dict = {k: [] for k in range(len(self.sections))}
        for idx1, a in enumerate(self.areas):
            for idx2, sarea in enumerate(self.sections):
                if sarea[0] ** 2 <= a < sarea[1] ** 2:
                    data_dict[idx2].append(self.max_iou_distribution.data[idx1])
        processes = [OneDimPlotter(value, str(key), cdf_pdf, axis_label=['iou', 'normalized numbers'],
                                   bins=10, range=(0, 1), ) for key, value in data_dict.items()]
        return Compose(processes, text=f'Iou distribution in range {self.sections}',
                       legend=[str(s) for s in self.sections])

    @property
    def anchor_coverage(self):
        """Distribution of each gt's anchor coverage."""
        if self.assigner is None:
            print(f"Assigner shouldn't be None.")
        return self._anchor_coverage_nums

    @anchor_coverage.setter
    def anchor_coverage(self, v):
        self._anchor_coverage_nums = OneDimPlotter(
            v, 'anchor coverage', cdf_pdf,
            axis_label=['anchor nums per gt', 'normalized numbers'],
            bins=20, range=(0, 40)
        )

    @property
    def section_anchor_coverage(self):
        """Distribution of each gt's anchor coverage in different sections."""
        if self.assigner is None:
            print(f"Assigner shouldn't be None.")
            return None
        data_dict = {k: [] for k in range(len(self.sections))}
        for idx1, a in enumerate(self.areas):
            for idx2, sarea in enumerate(self.sections):
                if sarea[0] ** 2 <= a < sarea[1] ** 2:
                    data_dict[idx2].append(self.anchor_coverage.data[idx1])
        processes = [
            OneDimPlotter(value, str(key), cdf_pdf,
                          axis_label=['anchor nums per gt', 'normalized numbers'],
                          bins=40, range=(0, 40)) for key, value in data_dict.items()]
        return Compose(processes, text=f'Anchor coverage in range {self.sections}',
                       legend=[str(s) for s in self.sections])
