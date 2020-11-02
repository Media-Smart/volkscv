from .coco_parse import COCOParser
from .image_parse import ImageParser
from .mmdet_parse import MMDETParser
from .txt_parse import TXTParser
from .xml_parse import XMLParser


def parse_data(format, need_shape=True, **kwargs):
    """An interface of data parser.

    Args:
        format (str): Data parser format.
        need_shape (bool): Whether need shape attributes. Default: True.

    Returns:
        obj (dict): Parsed data.

        Examples:

            {'img_names': ['COCO_val2014_000000581062.jpg'],
            'categories': ['person', ['bicycle']],
            'shapes': [[500 375]],
            'bboxes': [[[131.74  51.28 194.44 164.77],
                      [128.77 152.72 178.77 169.93]]],
            'labels': [[ 0 1]],
            'segs': [[[[134.12, 155.64, 136.9, 139.37]],
                    [[129.18, 160.51, 132.46, 159.07, 134.92, 163.17, 139.63]]]],
            'scores': None,
            'bboxes_ignore': [],
            'labels_ignore': [],}

    Examples:

        >>> anno = parse_data(format='coco',
        >>>                   need_shape=False,
        >>>                   ignore=True,
        >>>                   # txt_file='val2014/val.txt',
        >>>                   anno_path='val2014/instances_val2014.json',
        >>>                   imgs_folder='val2014',
        >>>                   )
        >>> print(anno)

    """

    if format.lower() == 'image':
        parse = ImageParser(**kwargs)
    elif format.lower() == 'coco':
        parse = COCOParser(**kwargs)
    elif format.lower() == 'xml':
        parse = XMLParser(**kwargs)
    elif format.lower() == 'mmdet':
        parse = MMDETParser(**kwargs)
    elif format.lower() == 'txt':
        parse = TXTParser(**kwargs)
    else:
        raise NotImplementedError(f"Currently support tasks are "
                                  f"['image', 'coco', 'xml', 'mmdet', 'txt'] "
                                  f"but got {format}")

    return parse(need_shape)
