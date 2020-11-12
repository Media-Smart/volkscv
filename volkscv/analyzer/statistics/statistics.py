from .processor import (BaseProcessor, ImageProcessor, LabelProcessor,
                        BoxProcessor, AnchorProcessor, DetFPProcessor)

PROCESSOR_DICT = dict(
    image=ImageProcessor,
    label=LabelProcessor,
    box=BoxProcessor,
    anchor=AnchorProcessor,
    fp=DetFPProcessor,
)


class DataStatistics(BaseProcessor):
    """ Analysis the distribution of (image size, image scale, image ratio, box scale, box size,
    box ratio, box per image, number of each class,. etc.)

    Args:
        data (dict): Parsed data using ``volkscv.parser.parser``.
        cfg (dict): Keys for different processors.

    Example:
        >>> data = dict(img_names=['./img/a.png'], labels=[[1]],
        >>>             shapes=[[800, 1333]], categories=['cat'],
        >>>             bboxes=[[[0, 0, 10, 10]]], segs=[[[0,0,1,1,2,2,3,3]]])
        >>> self = DataStatistics(data)
        >>> self.default_plot()
        >>> self.show()
        >>> self.export('./result', save_mode='pdf')
        >>> self.clear()
    """

    def __init__(self, data, cfg=None):
        super(DataStatistics, self).__init__(data)
        if cfg is None:
            cfg = dict(
                image=['img_names', 'labels', 'shapes', 'categories'],
                label=['labels', 'categories'],
                box=['bboxes', 'shapes', 'labels', 'categories'],
                seg=['segs', 'shapes', 'categories', 'labels'],
            )
        if isinstance(data, dict):
            for key, value in cfg.items():
                assert isinstance(value, list), f'type of value in cfg should' \
                                                f' be list but got {type(value)}'
                value = {v: data.get(v) for v in value}
                try:
                    tmp_p = PROCESSOR_DICT[key](value)
                except KeyError:
                    print(f'support keys are {PROCESSOR_DICT.keys()} '
                          f'but got {key}')
                    continue
                else:
                    self.processor.append(key)
                    setattr(self, key, tmp_p)


class AnchorStatistics(BaseProcessor):
    """ Analysis the bbox with anchor.

    Args:
        data (dict): Parsed data using ``volkscv.parser.parser``.
        anchor_generator (volkscv.analysis.utils.BaseAnchorGenerator): Anchor generator.
        target_shape (tuple): Image shape after resize. Current resize method is mmdetection version.
        assigner (volkscv.analysis.utils.BaseAssigner): Assigner is used to assign anchors to different gt bboxes.

    Example:
        >>> import numpy as np
        >>> from volkscv.analyzer.statistics.utils import AnchorGenerator, MaxIoUAssigner
        >>> data = dict(img_names=['./img/a.png'], labels=[[1]],
        >>>             shapes=[[800, 1333]], categories=['cat'],
        >>>             bboxes=[[[0, 0, 10, 10]]], segs=[[[0,0,1,1,2,2,3,3]]])
        >>> self = AnchorStatistics(data, (800, 1333), AnchorGenerator, MaxIoUAssigner)
        >>> self.default_plot()
        >>> self.show()
        >>> self.export('./result', save_mode='folder')
        >>> self.clear()
    """

    def __init__(self, data, target_shape, anchor_generator, assigner=None, device='cuda'):
        super(AnchorStatistics, self).__init__(data)
        self.processor = ['anchor']
        setattr(self, 'anchor', PROCESSOR_DICT['anchor'](data, target_shape, anchor_generator,
                                                         assigner, device))


class FPAnalysis(BaseProcessor):
    """ Analysis the bbox with anchor.

    Args:
        pred_json (str): Predict json file of coco format.
        gt_json (str): Ground truth json file of coco format.

    Example:
        >>> import json
        >>> import numpy as np
        >>> from volkscv.analyzer.statistics.utils import AnchorGenerator, MaxIoUAssigner
        >>> fake_pd = [{"image_id": 1, "bbox": [0, 0, 5, 5], "score": 0.5, "category_id": 1}]
        >>> fake_gt =  { "info": {}, "licenses": [],
        >>>              "images": [{ "license": 0, "file_name": 1, "coco_url": "",
        >>>                           "height": 500, "width": 400, "date_captured": "",
        >>>                           "flickr_url": "", "id": 1},],
        >>>              "annotations": [{"segmentation": [],"area": 100,"iscrowd": 0,"image_id": 1,
        >>>                               "bbox": [0, 0, 10, 10],"category_id": 1,"id": 99,"style": 0,"pair_id": 0,},],
        >>>              "categories": [ {"id": 1, "name": "1", "supercategory": "tt"},
        >>>                              {"id": 2, "name": "2", "supercategory": "tt"},
        >>>                              {"id": 3, "name": "3", "supercategory": "tt"},
        >>>                              {"id": 4, "name": "4", "supercategory": "tt"},]
        >>>             }
        >>> json.dump(fake_pd, open("fake_predict.json", 'a+'))
        >>> json.dump(fake_gt, open("fake_ground_truth.json", 'a+'))
        >>> self = FPAnalysis("fake_predict.json", "fake_ground_truth.json")
        >>> self.default_plot()
        >>> self.show()
        >>> self.export('./result', save_mode='folder')
        >>> self.clear()
    """

    def __init__(self, pred_json, gt_json):
        super(FPAnalysis, self).__init__({'pred': pred_json, 'gt': gt_json})
        self.processor = ['fp']
        setattr(self, 'fp', PROCESSOR_DICT['fp'](pred_json, gt_json))


def statistic_data(data, cfg=None):
    return DataStatistics(data, cfg)


def statistic_anchor(data, target_shape, anchor_generator,
                     assigner=None, device='cuda'):
    return AnchorStatistics(data, target_shape, anchor_generator, assigner, device)


def statistic_fp(pred_json, gt_json):
    return FPAnalysis(pred_json, gt_json)
