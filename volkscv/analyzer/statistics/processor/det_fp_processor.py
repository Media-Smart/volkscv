import copy

import matplotlib.pyplot as plt
import numpy as np
from volkscv.utils.cocoapi.pycocotools.coco import COCO
from volkscv.utils.cocoapi.pycocotools.cocoeval import COCOeval

from .base import BaseProcessor
from ..plotter import TwoDimPlotter, Compose


class DetFPProcessor(BaseProcessor):
    """ Process information of predict json and ground truth json.

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
        >>> self = DetFPProcessor("fake_predict.json", "fake_ground_truth.json")
        >>> self.default_plot()
        >>> self.show()
        >>> self.export('./result', save_mode='folder')
        >>> self.clear()
        >>> # specify class
        >>> self.fp_which_class([0]).plot()
        >>> self.export('./result', save_mode='folder')
        >>> self.clear()
    """

    def __init__(self, pred_json, gt_json):
        super(DetFPProcessor, self).__init__({'pred': pred_json, 'gt': gt_json})
        coco = COCO(gt_json)
        cocoDT = coco.loadRes(pred_json)
        self.cocoeval = COCOeval(coco, cocoDT, 'bbox')
        self.superclass = {k: [k] for k in range(1, len(self.cocoeval.cocoGt.cats) + 1)}
        self.processor = ['fp']

    def first_stage(self):
        # AP@iou[0.75, 0.5, 0.1], 0.1 means ignore loc problem.
        self.cocoeval.params.iouThrs = np.array([0.75, 0.5, 0.1])
        self.cocoeval.evaluate()
        self.cocoeval.accumulate()
        self.precision = self.cocoeval.eval['precision']
        self.precision[self.precision == -1] = 0

        return self.precision

    def second_stage(self, cat_ids=None):
        results = {}
        if cat_ids is None:
            cat_ids = list(range(1, 1 + len(self.cocoeval.cocoGt.cats)))
        for cat_id in cat_ids:
            temp_cocoeval = copy.deepcopy(self.cocoeval)
            temp_cocoeval.params.iouThrs = np.array([0.1])
            temp_cocoeval.params.useCats = True
            # 1. specify class
            temp_cocoeval.params.catIds = [cat_id]
            # 2. compute precision but ignore superclass confusion
            for k, v in temp_cocoeval.cocoGt.anns.items():
                if v['category_id'] in self.superclass[cat_id] and v['category_id'] != cat_id:
                    v['category_id'] = cat_id
                    v['iscrowd'] = 1

            # 3. sim
            temp_cocoeval.evaluate()
            temp_cocoeval.accumulate()
            prs_sim = temp_cocoeval.eval['precision']
            prs_sim[prs_sim == -1] = 0

            # 4. other: compute precision but ignore any class confusion
            for k, v in temp_cocoeval.cocoGt.anns.items():
                if v['category_id'] != cat_id:
                    v['category_id'] = cat_id
                    v['iscrowd'] = 1
            temp_cocoeval.evaluate()
            temp_cocoeval.accumulate()
            other_prs = temp_cocoeval.eval['precision']
            other_prs[other_prs == -1] = 0

            # 5. fill in background and false negative errors and plot
            bg = np.zeros_like(other_prs)
            bg[other_prs > 0] = 1
            fn = np.ones_like(other_prs)
            results[cat_id] = [prs_sim, other_prs, bg, fn]

        return results

    @property
    def fp(self):
        fres = self.first_stage()
        tres = self.second_stage()
        compose_pp = []
        legend = ['.75', '.5', 'Loc', 'Sim', 'Other', 'BG', 'FN']
        for idx in range(1, 1 + len(self.superclass)):
            c_fres = tres[idx]
            c_tres75 = fres[0, :, idx - 1, 0, -1]
            c_tres5 = fres[1, :, idx - 1, 0, -1]
            c_tres1 = fres[2, :, idx - 1, 0, -1]
            data = [c_tres75, c_tres5, c_tres1, c_fres[0][0, :, 0, 0, -1], c_fres[1][0, :, 0, 0, -1],
                    c_fres[2][0, :, 0, 0, -1], c_fres[3][0, :, 0, 0, -1]]
            pp = [TwoDimPlotter([np.arange(0.0, 1.01, 0.01), d], text=legend[idx2], func=plt.plot,
                                axis_label=['recall', 'precision'])
                  for idx2, d in enumerate(data)]
            compose_pp.append(Compose(pp, text=f'Analysis fp of class {idx}',
                                      legend=legend))

        return Compose(compose_pp, text='TotalResult', flag=True)

    def fp_specify_class(self, specify_class):
        assert isinstance(specify_class, list)
        fres = self.first_stage()
        tres = self.second_stage(specify_class)
        compose_pp = []
        legend = ['.75', '.5', 'Loc', 'Sim', 'Other', 'BG', 'FN']
        for idx in range(1, 1 + len(self.superclass)):
            c_fres = tres[idx]
            c_tres75 = fres[0, :, idx - 1, 0, -1]
            c_tres5 = fres[1, :, idx - 1, 0, -1]
            c_tres1 = fres[2, :, idx - 1, 0, -1]
            data = [c_tres75, c_tres5, c_tres1, c_fres[0][0, :, 0, 0, -1], c_fres[1][0, :, 0, 0, -1],
                    c_fres[2][0, :, 0, 0, -1], c_fres[3][0, :, 0, 0, -1]]
            pp = [TwoDimPlotter([np.arange(0.0, 1.01, 0.01), d], text=legend[idx2], func=plt.plot,
                                axis_label=['recall', 'precision'], )
                  for idx2, d in enumerate(data)]
            compose_pp.append(Compose(pp, text=f'Analysis fp of class {idx}',
                                      legend=legend))

        return Compose(compose_pp, text='TotalResult', flag=True)
