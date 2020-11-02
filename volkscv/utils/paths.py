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
        for line in fd:
            ll = line.strip().split()
            fnames.append(ll[0])
            if len(ll) > 1:
                annos.append(ll[1:])

    return fnames, annos
