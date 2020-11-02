# 1. Introduction
``Volkscv.analyzer.parse`` is designed for parsing annotation in different format, includes COCO, XML, TXT, result file format from mmdetection, ImageFolder format from trochvision.


# 2. Support
- [x] Parse img_names, categories, shapes, bboxes, labels, segs, bboxes_ignore, labels_ignore if existing

- [x] Support for partial data parsing 

- [x] Annotation format convert (xml2coco, coco2xml)


# 3. Usage
```python
from volkscv.analyzer.parse import parse_data, coco2xml_convert, xml2coco_convert


def test_coco():
    anno = parse_data(format='coco',
                      ignore=True,
                      # txt_file='val2014/val.txt',
                      anno_path='val2014/instances_val2014.json',
                      imgs_folder='val2014')
    print(anno)


def test_image():
    anno = parse_data(format='image',
                      #txt_file='data/val.txt',
                      imgs_folder='data/')
    print(anno)


def test_txt():
    anno = parse_data(format='txt',
                      txt_file='data/val.txt',
                      imgs_folder='data/',
                      categories=('1', '2',))
    print(anno)


def test_xml():
    anno = parse_data(format='xml',
                      ignore=True,
                      txt_file='data/val.txt',
                      imgs_folder='face/face',
                      xmls_folder='face/Annotations',
                      categories=('face',))
    print(anno)


def test_mmdet():
    anno = parse_data(format='mmdet',
                      anno_path='face/face.pkl.bbox.json',
                      # txt_file='face/val_.txt',
                      imgs_folder='face/face',
                      categories=('face',))
    print(anno)

def test_coco2xml():
    coco2xml_convert(anno_file='instances_val2014_.json',
                     output_dir='./result')


def test_xml2coco():
    xml2coco_convert(xml_list='data/overfit.txt',
                     xml_dir='data/Annotations',
                     json_file='./output.json')



if __name__ == '__main__':
    test_coco()
    # test_image()
    # test_txt()
    # test_xml()
    # test_mmdet()
    # test_coco2xml()
    # test_xml2coco()
```