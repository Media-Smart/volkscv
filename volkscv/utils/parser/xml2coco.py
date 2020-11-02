import json
import os
import xml.etree.ElementTree as ET

PRE_DEFINE_CATEGORIES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor')


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError(f'can not find {name} in {root.tag}')
    if 0 < length != len(vars):
        raise NotImplementedError(
            f'The size of {name} is supposed to be {length}, but is {len(vars)}')
    if length == 1:
        vars = vars[0]
    return vars


def xml2coco_convert(xml_list,
                     xml_dir,
                     categories=PRE_DEFINE_CATEGORIES,
                     json_file='./output.json',
                     start_cate_id=1):
    """converting XML format to COCO format json.

    Args:
        xml_list (str): Path to annotation files ids list.
        xml_dir (str): Path to annotation files directory.
        categories (list or tuple): Categories.
        json_file (str): Path to output json file.
        start_cate_id (int): Start id of categories. Default: 1.
    """

    list_fp = open(xml_list, 'r')
    json_dict = {'images': [], 'type': 'instances', 'annotations': [],
                 'categories': []}
    anno_id = 0
    image_id = 0
    categories = {name: i + start_cate_id for i, name in enumerate(categories)}
    for line in list_fp:
        line = f'{line.strip()}.xml'
        print(f'Processing {line}')
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = root.findall('path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError(
                f'{len(path)} paths found in {line}')
        # The filename must be a number
        image_id += 1
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        #  Currently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in root.findall('object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            anno_id += 1
            ann = {'area': o_width * o_height, 'iscrowd': 0,
                   'image_id': image_id,
                   'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': anno_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()
    print(f'convert json file {json_file} done!')
