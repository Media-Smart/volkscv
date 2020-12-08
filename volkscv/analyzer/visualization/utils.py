import os
from collections import deque

import cv2
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from ...utils.geos import bbox_overlaps
from ...utils.paths import read_imglist


def save_img(save_path, img):
    """Saves an image to a specified file.

    Args:
        save_path (str): Name of the file.
        img (numpy.ndarray): Image to be saved.
    """

    if os.path.exists(save_path):
        save_path = save_path[:-4]
        save_path += '_new.png'
    cv2.imwrite(save_path, img)


def get_index_list(specified_imgs, default_index_list, extension=('all',)):
    """Get index list for image browsing.

    Args:
        specified_imgs (str): Images need to be viewed, folder or txt file.
        default_index_list (list): Default images.
        extension (str): Image extension. Default: 'jpg'.

    Returns:
        (deque): Images fname list.
    """

    if not isinstance(extension, tuple):
        extension = (extension, )
    if specified_imgs is None:
        index_list = deque(default_index_list)
    elif os.path.isdir(specified_imgs):
        img_list = os.listdir(specified_imgs)
        index_list = deque([os.path.join(specified_imgs, i) for i in img_list if i.split('.')[-1] in extension])
    elif specified_imgs.endswith('txt'):
        img_list, _ = read_imglist(specified_imgs)
        if 'all' in extension:
            index_list = deque(img_list)
        else:
            index_list = deque(
                [i for i in img_list if i.split('.')[-1] in extension])
    else:
        index_list = deque(default_index_list)
    return index_list


def show_img(img):
    """show img using OpenCV.

    Args:
        img (numpy.ndarray): Image being shown.

    Returns:
        (int): Index of pressed key.
    """

    cv2.namedWindow('visualization', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('visualization',
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow('visualization', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def get_pallete(categories):
    """Generate pallete for categories.

    Args:
        categories (list or tuple): Names of all categories.

    Returns:
        dict: Pallete of all categories.
    """

    num_cls = len(categories) + 1
    color_map = num_cls * [0, 0, 0]
    for i in range(0, num_cls):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    pallete = dict()
    for idx, i in enumerate(range(3, len(color_map), 3)):
        pallete.update({categories[idx]: color_map[i:i + 3]})

    return pallete


def adjust_color_brightness(color, brightness_factor=0.5):
    """Adjust color brightness.

    Args:
        color (numpy.ndarray): Color.
        brightness_factor (float): Factor to adjust color brightness.

    Returns:
        numpy.ndarray: Modified_color.
    """

    img_bgr = np.array(color)[np.newaxis, np.newaxis, :]
    img_bgr = img_bgr.astype(np.uint8)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hsv[..., 2] = img_hsv[..., 2] * brightness_factor
    modified_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR).squeeze()
    return modified_color


def generate_legend(colors, categories):
    """Generate legend for visualization, to show labels and corresponding color.

    Args:
        colors: (np.ndarray): Colors, corresponding to category.
        categories (set): Categories need to be displayed.

    Returns:
        list: Legend.
    """

    red_patch = [
        mpatches.Patch(
            color=np.array(
                [colors[cate][2], colors[cate][1], colors[cate][0]]) / 255,
            label=cate) for cate in list(categories)]
    return red_patch


def generate_mpl_figure(imgs, fname, title, legend=None):
    """Generate matplotlib figure.

    Args:
        imgs (dict): Images of ori, gt or pred.
        fname (str): Figure title.
        title (dict): Subfigure titles.
        legend (list): Legend of figure.

    Returns:
        numpy.ndarray: Image.
    """

    fig, axs = plt.subplots(1, len(imgs), figsize=(12, 5), sharex='col',
                            sharey='row', constrained_layout=True, dpi=400)
    fig.suptitle(f"visualization: {fname.split('/')[-1]}", fontsize=12)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    for idx, (k, v) in enumerate(imgs.items()):
        axs[idx].set_title(title[k])
        axs[idx].imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))

    if legend is not None:
        fig.legend(handles=legend, prop={'size': 6}, loc='upper left')

    canvas = fig.canvas
    s, (width, height) = canvas.print_to_buffer()
    img = Image.frombytes("RGBA", (width, height), s)

    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.close()
    return img


def draw_image(img,
               key,
               data,
               categories,
               show_score=True,
               ):
    """Draw image for classification task.

    Args:
        img (numpy.ndarray): Image.
        key (str): Info of Image, includes ori, gt and pred.
        data (dict): Annotation of current image.
        categories (list): Categories of whole dataset.

    Returns:
        img_ (numpy.ndarray): Drawn image.
        text (str): labels and scores' information.
    """

    img_ = img.copy()
    h, w, _ = img_.shape
    anno = data[key]
    label = anno.get('labels')
    score = anno.get('scores')

    label = categories[label]

    text = f'label: {label}'
    if show_score and score is not None:
        text += f', score: {score}'
    return img_, text


def cal_seg_fpfn(data,
                 shape,
                 colors,
                 categories,
                 category_to_show=None):
    """Calculate fp and fn for segmentation.

    Args:
        data (dict): Annotation of current image.
        shape (tuple): Shape of image.
        colors (dict): Color for different image.
        categories (list): Categories of whole dataset.
        category_to_show (tuple): Categories need to be displayed.

    Returns:
        fp_mask (numpy.ndarray): False Positive mask for pred.
        fn_mask (numpy.ndarray): False Negative mask for gt.

    """

    assert 'gt' in data.keys() and 'pred' in data.keys()
    gt = data['gt']
    pred = data['pred']
    gt_seg = gt.get('segs', np.array([]))
    gt_labels = gt.get('labels', np.array([]))
    dt_seg = pred.get('segs', np.array([]))
    dt_labels = pred.get('labels', np.array([]))

    assert gt_seg.shape[0] == gt_labels.shape[0]
    assert dt_seg.shape[0] == dt_labels.shape[0]

    gt_mask = np.zeros(shape, dtype=np.uint8)
    pred_mask = np.zeros(shape, dtype=np.uint8)

    for seg, label in zip(dt_seg, dt_labels):
        class_label = categories[label]
        if category_to_show is None or class_label in category_to_show:
            for m in seg:
                m = np.array(m).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(pred_mask, [m], colors[class_label])

    for seg, label in zip(gt_seg, gt_labels):
        class_label = categories[label]
        if category_to_show is None or class_label in category_to_show:
            for m in seg:
                m = np.array(m).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(gt_mask, [m], colors[class_label])

    if gt_seg.shape[0] == 0 or dt_seg.shape[0] == 0:
        return pred_mask, gt_mask

    tp_mask = gt_mask & pred_mask
    fp_mask = pred_mask - tp_mask
    fn_mask = gt_mask - tp_mask

    return fp_mask, fn_mask


def cal_det_fpfn(data,
                 categories,
                 iou_thr=0.5,
                 score_thr=-1,
                 category_to_show=None):
    """Calculate fp and fn for detection.

    Args:
        data (dict): Annotation of current image.
        categories (list): Categories of whole dataset.
        iou_thr (float): IoU threshold to be considered as matched for bboxes.
            Default: 0.5.
        score_thr (float): Minimum score of bboxes to be shown. Default: -1.
        category_to_show (tuple or None): Categories need to be displayed.


    Returns:
        tuple (fp, tp, fn, gt_covered) whose elements are True and False.
    """

    assert 'gt' in data.keys() and 'pred' in data.keys(), \
        'data should have both gt and pred'
    gt = data['gt']
    pred = data['pred']
    gt_bboxes = gt.get('bboxes', np.zeros((0, 4)))
    gt_labels = gt.get('labels', np.zeros((0,)))
    gt_labels_ignore = gt.get('labels_ignore', np.zeros((0,)))
    gt_bboxes_ignore = gt.get('bboxes_ignore', np.zeros((0, 4)))

    pred_bboxes = pred.get('bboxes', np.zeros((0, 4)))
    pred_labels = pred.get('labels', np.zeros((0,)))
    scores = pred.get('scores', np.zeros((0,)))

    if scores is not None:
        if score_thr > 0:
            assert pred_bboxes.shape[1] == 4
            inds = scores > score_thr
            pred_bboxes = pred_bboxes[inds, :]
            pred_labels = pred_labels[inds]
            scores = scores[inds]

    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))

    gt_bboxes = np.concatenate((gt_bboxes, gt_bboxes_ignore), axis=0)
    gt_labels = np.concatenate((gt_labels, gt_labels_ignore), axis=0)

    assert gt_bboxes.shape[0] == gt_labels.shape[0]
    assert pred_bboxes.shape[0] == pred_labels.shape[0]

    fp = np.zeros(pred_bboxes.shape[0], dtype=np.bool)
    tp = np.zeros(pred_bboxes.shape[0], dtype=np.bool)
    gt_covered = np.zeros(gt_bboxes.shape[0], dtype=np.bool)
    fn = np.zeros(gt_bboxes.shape[0], dtype=np.bool)

    if category_to_show is None:
        categories = range(len(categories))
    else:
        categories = [categories.tolist().index(c) for c in category_to_show]

    for i in categories:
        gt_mask = gt_labels == i
        pred_mask = pred_labels == i
        gt_bbox = gt_bboxes[gt_mask]
        pred_bbox = pred_bboxes[pred_mask]
        score = scores[pred_mask]
        gt_ignore_ind = gt_ignore_inds[gt_mask]
        if sum(gt_mask) > 0 or sum(pred_mask) > 0:
            fp_, tp_, fn_, gt_covered_ = fpfn(gt_bbox, pred_bbox, score,
                                              gt_ignore_ind,
                                              iou_thr=iou_thr)

            fp[pred_mask] = fp_
            gt_covered[gt_mask] = gt_covered_
            tp[pred_mask] = tp_
            fn[gt_mask] = fn_

    return fp, tp, fn, gt_covered


def fpfn(gt_bbox,
         pred_bbox,
         confidence,
         gt_ignore_ind,
         iou_thr=0.5):
    """Calculate fp and fn for detection for specified category.

    Args:
        gt_bbox (numpy.ndarray): GT bboxes of this image, shape (n, 4).
        pred_bbox (numpy.ndarray): Predicted bboxes of this image, shape (m, 5).
        confidence (numpy.ndarray): Scores of this image, shape (m, ).
        gt_ignore_ind (numpy.ndarray): An indicator of ignored gts.
        iou_thr (float): IoU threshold to be considered as matched for bboxes.
            Default: 0.5.

    Returns:
        tuple: (fp, tp, fn, gt_covered) whose elements are True and False.
    """

    num_dets = pred_bbox.shape[0]
    num_gts = gt_bbox.shape[0]

    tp = np.zeros(num_dets, dtype=np.bool)
    fp = np.zeros(num_dets, dtype=np.bool)
    gt_covered = np.zeros(num_gts, dtype=np.bool)
    fn = ~gt_ignore_ind

    if num_gts == 0:
        fp[...] = True
        return fp, tp, fn, gt_covered

    if num_dets == 0:
        return fp, tp, fn, gt_covered

    ious = bbox_overlaps(torch.from_numpy(pred_bbox), torch.from_numpy(gt_bbox))
    ious_max = ious.numpy().max(axis=1)
    ious_argmax = ious.numpy().argmax(axis=1)
    sort_inds = np.argsort(-confidence)

    for i in sort_inds:
        if ious_max[i] >= iou_thr:
            matched_gt = ious_argmax[i]
            if not gt_ignore_ind[matched_gt]:
                if not gt_covered[matched_gt]:
                    fn[matched_gt] = False
                    gt_covered[matched_gt] = True
                    tp[i] = True
                else:
                    fp[i] = True
        else:
            fp[i] = True

    return fp, tp, fn, gt_covered


def draw_bbox(img,
              key,
              data,
              colors,
              categories,
              category_to_show=None,
              show_score=False,
              show_fpfn=False,
              show_fpfn_format='line',
              show_ignore=False,
              score_thr=0.3,
              base_thickness=1,
              base_fontscale=0.5,
              **kwargs):
    """Draw image for detection task.

    Args:
        img (numpy.ndarray): Image.
        key (str): Info of Image, includes ori, gt and pred.
        data (dict): Annotation of current image.
        colors (dict): Color for different image.
        categories (list): Categories of whole dataset.
        category_to_show(tuple or None): Categories need to be displayed.
        show_score (bool): Whether put score onto image. Default: False.
        show_fpfn (bool): Whether show fp and fn. Default: False.
        show_fpfn_format (str): Display form of fp and fn.
        show_ignore (bool): Whether show ignore box and matched with ignore. Default: False.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.3.
        base_thickness (int): Default thickness of lines. Default: 1.
        base_fontscale (float): Font scale factor. Default: 0.5.

    Returns:
        (numpy.ndarray): Drawn image.
    """

    img_ = img.copy()
    anno = data[key]
    if not anno:
        return img_, None

    bboxes = anno.get('bboxes')
    labels = anno.get('labels')
    scores = anno.get('scores')
    not_ignore = bboxes.shape[0]
    labels_ignore = anno.get('labels_ignore', np.zeros((0,)))
    bboxes_ignore = anno.get('bboxes_ignore', np.zeros((0, 4)))

    bboxes = np.concatenate((bboxes, bboxes_ignore), axis=0)
    labels = np.concatenate((labels, labels_ignore), axis=0).astype(np.int32)

    assert show_fpfn_format in ['line', 'mask']
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert scores is None or scores.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4

    masks = np.zeros(img_.shape, dtype=np.uint8)
    f, t = [], []
    if scores is not None:
        if score_thr > 0:
            assert bboxes.shape[1] == 4
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]

    if show_fpfn:
        if key == 'gt':
            t = kwargs['tn']
            f = kwargs['fn']
        else:
            t = kwargs['tp']
            f = kwargs['fp']

    color_ignore = colors.get('ignore', (0, 0, 255))
    color_matched_with_ignore = colors.get('matched_with_ignore',
                                           (0, 0, 255))
    color_text = colors.get('text', (0, 0, 255))

    for idx in range(bboxes.shape[0]):
        bbox = bboxes[idx]
        label = labels[idx]
        class_label = categories[label]
        color = colors[class_label]
        if category_to_show is None or class_label in category_to_show:
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])

            text = ''
            if scores is not None:
                text += f'{scores[idx]:.3f}'
            if show_score:
                area = (bbox_int[2] - bbox_int[0]) * (bbox_int[3] - bbox_int[1])
                fontscale = base_fontscale * np.sqrt(area) / 80
                cv2.putText(img_, text, (bbox_int[0], bbox_int[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=fontscale,
                            color=color_text)

            if not show_fpfn:
                cv2.rectangle(img_, left_top, right_bottom,
                              color=color, thickness=base_thickness)
                if show_ignore and idx >= not_ignore:
                    thickness = base_thickness
                    color = color_ignore
                    cv2.rectangle(
                        img_, left_top, right_bottom, color=color,
                        thickness=thickness)
            else:
                if len(f) and f[idx]:
                    if show_fpfn_format == 'mask':
                        cv2.rectangle(masks, left_top, right_bottom,
                                      color=color, thickness=-1)
                        continue
                    else:
                        thickness = base_thickness * 2
                elif len(t) and t[idx]:
                    thickness = base_thickness
                else:
                    if show_ignore:
                        thickness = base_thickness
                        color = color_ignore if key == 'gt' \
                            else color_matched_with_ignore
                    else:
                        continue
                cv2.rectangle(
                    img_, left_top, right_bottom, color=color,
                    thickness=thickness)

    if show_fpfn and show_fpfn_format == 'mask':
        img_ = cv2.addWeighted(img_, 1, masks, 0.5, 0)

    return img_, None


def poly2bbox(poly):
    """Draw image for detection task.

    Args:
        poly (numpy.ndarray): Polygon of segmentation mask.

    Returns:
        (list): Bbox coordinates.
    """

    poly = poly.reshape(-1, 2)
    x_coordinates, y_coordinates = zip(*poly)

    return [int(min(x_coordinates)), int(min(y_coordinates)),
            int(max(x_coordinates)), int(max(y_coordinates))]


def draw_mask(img,
              key,
              data,
              colors,
              categories,
              category_to_show=None,
              show_score=False,
              show_fpfn=False,
              score_thr=-1,
              base_fontscale=0.5,
              **kwargs):
    """Draw image for segmentation task.

    Args:
        img (numpy.ndarray): Image.
        key (str): Info of Image, includes ori, gt and pred.
        data (dict): Annotation of current image.
        colors (dict): Color for different image.
        categories (list): Categories of whole dataset.
        category_to_show(tuple or None): Categories need to be displayed.
        show_score (bool): Whether put score onto image. Default: False.
        show_fpfn (bool): Whether show fp and fn. Default: False.
        score_thr (float): Minimum score of bboxes to be shown. Default: -1.
        base_fontscale (float): Font scale factor. Default: 0.5.

    Returns:
        (numpy.ndarray): Drawn image
    """

    img_ = img.copy()
    anno = data[key]
    if not anno:
        return img_, None

    segs = anno.get('segs')
    labels = anno.get('labels')
    scores = anno.get('scores')

    assert labels.ndim == 1
    assert segs.shape[0] == labels.shape[0]

    masks = np.zeros(img_.shape, dtype=np.uint8)
    fpfnmask = np.ones(img_.shape, dtype=np.bool)
    if scores is not None:
        if score_thr > 0:
            inds = scores > score_thr
            segs = segs[inds, :]
            labels = labels[inds]
            scores = scores[inds]

    if show_fpfn:
        fpfnmask = kwargs['fn'] if key == 'gt' else kwargs['fp']

    color_text = colors.get('text', (0, 0, 255))

    for idx in range(segs.shape[0]):
        seg = segs[idx]
        label = labels[idx]
        class_label = categories[label]
        color = colors[class_label]

        if category_to_show is None or class_label in category_to_show:
            for m in seg:
                m = np.array(m).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(masks, [m], color)

                text = ''
                if scores is not None:
                    text += f'{scores[idx]:.3f}'
                if show_score:
                    bbox_int = poly2bbox(m)
                    area = (bbox_int[2] - bbox_int[0]) * (
                            bbox_int[3] - bbox_int[1])
                    fontscale = base_fontscale * np.sqrt(area) / 80
                    cv2.putText(img_, text,
                                (bbox_int[0], bbox_int[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=fontscale,
                                color=color_text)

    img_ = cv2.addWeighted(img_, 1, masks, 0.5, 0)
    if show_fpfn:
        img_ = cv2.addWeighted(img_, 0.8, fpfnmask, 1, 0)

    return img_, None
