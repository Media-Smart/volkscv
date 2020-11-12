import json
import os

import cytoolz
from lxml import etree, objectify


def instance2xml_base(anno):
    """Parse xml base information from annotation.
    
    Args:
        anno (dict): Basic annotatio of image.

    Returns:
        object: Anno tree.
    """

    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC dataset'),
        E.filename(anno['file_name']),
        E.source(
            E.database('COCO format'),
            E.annotation('COCO format'),
            E.image('Flickr'),
        ),
        E.size(
            E.width(anno['width']),
            E.height(anno['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """Parse xml instance information from annotation.

    bbox_type: xyxy (xmin, ymin, xmax, ymax);
               xywh (xmin, ymin, width, height)

    Args:
        anno (dict): Instance annotatio of image.
        bbox_type (str): Format of bbox.

    Returns:
        object: Anno tree.
    """

    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin + w
        ymax = ymin + h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def coco2xml_convert(anno_file,
                     output_dir='./result',
                     folder_split=False):
    """converting COCO format to xml format.

        Args:
            anno_file (str): Path to annotations of data.
            output_dir (str): Path to save folder.
            folder_split (bool): Whether to store file by category.
        """

    os.makedirs(output_dir, exist_ok=True)
    annotation = json.load(open(anno_file, 'r'))
    categories = {d['id']: d['name'] for d in annotation['categories']}
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge,
                                cytoolz.join('id', annotation['images'],
                                             'image_id',
                                             annotation['annotations'])))
    # convert category id to name
    for instance in merged_info_list:
        instance['category_id'] = categories[instance['category_id']]
    # group by filename to pool all bbox in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        anno_tree = instance2xml_base(groups[0])
        filenames = []
        for group in groups:
            if folder_split:
                subdir = os.path.join(output_dir, group['category_id'])
                os.makedirs(subdir, exist_ok=True)
                filenames.append(
                    os.path.join(subdir, os.path.splitext(name)[0] + '.xml'))
            else:
                filenames = [
                    os.path.join(output_dir,
                                 os.path.splitext(name)[0] + '.xml')]

            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))
        for filename in filenames:
            etree.ElementTree(anno_tree).write(filename, pretty_print=True)
        print(f'Formating instance xml file {name} done!')
