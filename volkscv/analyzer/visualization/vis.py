from .classification import ClsVis
from .detection import DetVis
from .segmentation import SegVis

CFG = dict(
    cls=['img_names', 'labels', 'scores'],
    det=['img_names', 'labels', 'bboxes', 'scores', 'bboxes_ignore', 'labels_ignore'],
    seg=['img_names', 'labels', 'segs', 'scores'],
)


def visualization(task, **kwargs):
    """An interface of visualization.

    Args:
        task (str): Task name, cls, det or seg.

    Returns:
        obj (Vis)

     Examples:
        >>>gt_anno = dict(img_names=np.array(['data/example.jpg', ]),
        >>>               categories=np.array(['person', ]),
        >>>               shape=np.array([[1024, 768], ]),
        >>>               bboxes=np.array([[131.74, 51.28, 194.44, 164.77],
        >>>                                [128.77, 152.72, 178.77, 169.93]]),
        >>>               labels=np.array([0, 0]),
        >>>               segs=None,
        >>>               scores=None,
        >>>               bboxes_ignore=np.array([]),
        >>>               labels_ignore=np.array([]))
        >>>dt_anno = dict(img_names=np.array(['data/example.jpg', ]),
        >>>               categories=np.array(['person', ]),
        >>>               shape=np.array([[1024, 768], ]),
        >>>               bboxes=np.array([[50, 50, 150, 150],
        >>>                                [131.74, 51.28, 194.44, 164.77],
        >>>                                [128.77, 152.72, 178.77, 169.93],
        >>>                                [100, 100, 200, 200],]),
        >>>               labels=np.array([0, 0, 0, 0]),
        >>>               segs=None,
        >>>               scores=np.array([0.98,0.92, 0.75, 0.89]),
        >>>               bboxes_ignore=np.array([]),
        >>>               labels_ignore=np.array([]))
        >>> vis = visualization(task='det', gt=gt_anno, pred=pred_anno)
        >>> params = dict(save_folder='./result',
        >>>               category_to_show=('person',),
        >>>               show_score=False,
        >>>               show_fpfn=True,
        >>>               show_fpfn_format='line')
        >>>
        >>> vis.show(**params)
        >>> # vis.save(**params)

    """

    cfg = CFG[task]
    if task.lower() == 'cls':
        vis = ClsVis(cfg, **kwargs)
    elif task.lower() == 'det':
        vis = DetVis(cfg, **kwargs)
    elif task.lower() == 'seg':
        vis = SegVis(cfg, **kwargs)
    else:
        raise NotImplementedError(f"Currently support tasks are ['cls', 'det', seg] but got {task}")

    return vis
