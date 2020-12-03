import warnings
from abc import ABCMeta, abstractmethod

import numpy as np

from ...utils.cocoapi.pycocotools.coco import COCO
from ...utils.cocoapi.pycocotools.cocoeval import COCOeval

from ..base import BaseMetric


class BaseDetMetric(BaseMetric, metaclass=ABCMeta):
    """
    Base metric for detection metrics.
    This class is abstract, providing a standard interface for metrics of this type.
    """
    def __init__(self, cates_num=None, cates_ids=None, mode='file'):
        super().__init__()
        self.mode = mode
        if self.mode == 'online':
            warnings.warn('cates id start from 1, if ids of categories are not continuous, '
                          'param categories_id is needed!')
            self.cates_num = cates_num
            self.cates_ids = cates_ids
            if self.cates_ids:
                assert (type(self.cates_ids) is list) and (type(self.cates_ids[0]) is int), \
                    'Type error for cates_ids, a list of integers is needed!'
            else:
                assert self.cates_num is not None, 'Param cates_num is necessary for "online" mode!'

        self.iters = 0
        self.pred_accumulate = None
        self.target_accumulate = None

    @abstractmethod
    def compute(self, pred, target):
        """
        Compute metric value for current epoch.

        Args:
            In 'file' mode:
            pred: path to results json file, or a list.
                Prediction results from detection model, stored in a dict, following the format of COCO
                [{'image_id': XX, 'bbox': [x, y, w, h], 'score': X, 'category_id': X }, ...]
            target: path to ground truth file, or a dict.
                following the format of COCO annotation.
                {'info': {},
                 'licenses': [],
                 'images': [
                     {'file_name': X,
                      'height': X,
                      'width': X,
                      'id': X
                     },
                     ...
                 ],
                 'annotations':[
                     {'segmentation': [],
                      'area': X,
                      'iscrowd': 0 or 1,
                      'image_id': X,
                      'bbox': [x, y w, h],
                      'category_id': X,
                      'id': X,
                     },
                     ...
                 ],
                 'categories':[
                     {'id': X,
                      'name': X,
                      'supercategory': X,
                     },
                     ...
                 ]
                }
            In 'online' mode:
            pred: list
                Prediction results from detection model, stored in a list, with length of B = batchsize,
                each element of the list is a numpy.ndarray with shape of (N, 6),
                N is the number of bboxes, 6 stands for the information of box [x, y, w, h, score, cat_id]
            target: list
                Ground truth, stored in a list, with length of B = batchsize, each element of the list is
                a numpy.ndarray with shape of (N, 6), N is the number of bboxes, 6 stands for the information
                of box [x, y, w, h, cat_id, iscrowd]

        Returns:
            metric value for current epoch
        """
        pass

    def update(self, pred, target):
        self.format(pred=pred, target=target)
        self.iters += 1
        if self.pred_accumulate is None:
            self.pred_accumulate = self.pred_temp.copy()
        else:
            self.pred_accumulate = np.concatenate((self.pred_accumulate, self.pred_temp.copy()), axis=0)

        if self.target_accumulate is None:
            self.target_accumulate = self.target_temp.copy()
        else:
            self.target_accumulate['images'].extend(self.target_temp['images'].copy())
            self.target_accumulate['annotations'].extend(self.target_temp['annotations'].copy())

    def format(self, pred, target):
        assert type(pred) == type(target) == list, \
            "Pred and target need to be list under 'online' mode!"
        assert len(pred) == len(target) and pred[0].shape[-1] == target[0].shape[-1] == 6,\
            "pred & target do not match in shape!"

        image_id_start = self.iters*len(pred)
        # pred reformat
        pred_valid = []
        for i, pred_img in enumerate(pred):
            fake_image_id = np.ones((pred_img.shape[0], 1)) * (image_id_start + i + 1)
            pred_valid.extend(list(np.concatenate((fake_image_id, pred_img), axis=1)))
        self.pred_temp = np.array(pred_valid)
        # target reformat
        target_valid = {}

        if self.cates_ids:
            cates = [{'id': cate_id, 'name': cate_id} for cate_id in self.cates_ids]
        else:
            cates = [{'id': cate_id+1, 'name': cate_id+1} for cate_id in range(self.cates_num)]
        target_valid.update({'categories': cates})

        images_temp_info = [{'id': image_id_start + i + 1} for i in range(len(target))]
        target_valid.update({'images': images_temp_info})

        annotations = []
        j = 1
        for i, img_info in enumerate(target):
            for ann in img_info:
                ann_dict = {}
                ann_dict['image_id'] = image_id_start + i + 1
                ann_dict['bbox'] = ann[0:4]
                ann_dict['category_id'] = int(ann[4])
                ann_dict['iscrowd'] = int(ann[5])
                ann_dict['area'] = ann[2]*ann[3]
                ann_dict['id'] = (image_id_start + i + 1)*1000 + j
                annotations.append(ann_dict)
                j += 1
        target_valid.update({'annotations': annotations})
        self.target_temp = target_valid

    def check(self, pred, target):
        """
        Check inputs
        """
        self._check_type(pred, target)

    @staticmethod
    def _check_type(pred, target):
        assert (type(pred) in [str, list]) and (type(target) in [str, dict]), \
            "Format of inputs not supported! " \
            "pred need to be str or array & target need to be str or dict under 'file' mode!"

    def accumulate(self):
        assert self.mode == 'online', "accumulate() can only be used in 'online' mode!"
        accumulate_state = self.compute(self.pred_accumulate, self.target_accumulate)
        return accumulate_state

    def __call__(self, pred, target):
        self.check(pred, target)

        if self.mode == 'online':
            self.update(pred, target)
            self.compute(pred, target)
        elif self.mode == 'file':
            self.compute(pred, target)


class COCOAnalysis(BaseDetMetric):
    """
    Basic analysis using pycocotools.
    """
    def __init__(self,
                 iou=None,
                 maxdets=None,
                 areaRng=None,
                 areaRngLbl=None,
                 cates_num=None,
                 cates_ids=None,
                 mode='online'):
        super().__init__(cates_num=cates_num,
                         cates_ids=cates_ids,
                         mode=mode)
        self.iou = iou
        self.maxdets = maxdets
        self.areaRng = areaRng
        self.areaRngLbl = areaRngLbl

    def reset(self):
        self.cocoEval = None
        self.precision = None
        self.recall = None
        self.score = None

    def compute(self, pred, target):
        coco = COCO(target)
        cocoDT = coco.loadRes(pred)
        self.cocoEval = COCOeval(coco, cocoDT, 'bbox')
        self.cate_name = [cate['name'] for cate in coco.dataset['categories']]
        self._param_setter()
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.precision = self.cocoEval.eval['precision']
        self.recall = self.cocoEval.eval['recall']
        self.score = self.cocoEval.eval['scores']

    def _param_setter(self):
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
