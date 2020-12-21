def filter_imgs(bbox, min_size=None, format='xywh'):
    """Filter image acoording to box size.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field. Default: None.
        bbox (numpy.ndarray or list): Bouding box.
        format (str): Format of box.
    """

    if min_size is not None:
        if format == 'xyxy':
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        else:
            w = bbox[2]
            h = bbox[3]
        return w < min_size or h < min_size
    else:
        return False


def read_imglist(imglist_path):
    """Read content form a txt file.

    Args:
        imglist_path (str): Absolute path of txt file.

    Returns:
        fnames (list): Image names.
        annos (list): Annotations.
    """

    fnames, annos = [], []
    with open(imglist_path, 'r') as fd:
        for line in fd.readlines():
            ll = line.strip().split()
            fnames.append(ll[0])
            if len(ll) > 1:
                annos.append(ll[1:])

    return fnames, annos
