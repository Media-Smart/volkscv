import os
import random
import time
from abc import abstractmethod, ABCMeta

import numpy as np

from .utils import show_img, get_pallete, get_index_list, save_img


class BaseVis(metaclass=ABCMeta):
    """The base class of visualization.

    All subclasses should implement the following APIs:

    - ``img_process()``

    Args:
        cfg (list or tuple): The data may used in this task.
        gt (dict): Gt data for visualization.
        pred (dict): Pred data for visualization.
        colors (dict): Colors for visualization.
        extension (str): Image extention. Default: 'jpg'.
    """

    def __init__(self,
                 cfg,
                 gt=None,
                 pred=None,
                 colors=None,
                 extension='jpg',
                 ):
        super(BaseVis, self).__init__()
        self.cfg = cfg
        self.data = dict(gt=gt, pred=pred)
        self.extension = extension
        self.default_title = dict(ori='ori_img',
                                  gt='img_with_gt',
                                  pred='img_with_pred')

        for key in list(self.data.keys()):
            if self.data[key] is not None:
                attr = self.data[key]
                self.set('img_names', attr.get('img_names'))
                self.set('categories', attr.get('categories'))
                for k in list(attr.keys()):
                    if k not in self.cfg:
                        attr.pop(k)
            else:
                self.data.pop(key)

        self.check()

        self._colors = get_pallete(
            self.categories) if colors is None else colors
        self.fnames = [os.path.split(i)[-1] for i in self.img_names]
        self.img_prefixs = [os.path.split(i)[0] for i in self.img_names]

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, v):
        self._colors.update(v)

    def check(self):
        """check attributes."""

        assert hasattr(self, 'img_names'), \
            "Keys of input data should contain 'img_names'."
        assert hasattr(self, 'categories'), \
            "Keys of input data should contain 'categories.'"
        assert self.data, \
            "Keys of input data should contain at least one item in 'gt' or 'pred'."

    def set(self, name, value):
        """Set the named attribute on the given object to the specified value."""

        if not hasattr(self, name):
            return setattr(self, name, value)

    def show(self,
             save_folder=None,
             specified_imgs=None,
             shuffle=False,
             **kwargs):
        """Display data dynamically.

        The function supports interaction,
        press key 's' to save the current picture,
        press key 'b' to go back to the previous picture, cache latest 32 images,
        press key 'q' to quit browse mode, press rest keys to go into next image.

        Args:
            save_folder (str, optional): Folder path where the images may store.
            specified_imgs: (str, optional): Specified visualization to show, folder or txt file.
            shuffle: (bool, optional): Shuffle the dataset. default: False.
        """

        if save_folder is None:
            save_folder = f"./{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(save_folder, exist_ok=True)

        index_list = get_index_list(specified_imgs, self.fnames, self.extension)

        if shuffle:
            random.seed(1)
            random.shuffle(index_list)

        current_show = -1
        key = -1
        while len(index_list):
            fname = index_list[0]
            fname = os.path.join(self.img_prefixs[self.fnames.index(fname)],
                                 fname)
            img, flag = self.img_process(fname, **kwargs)
            if flag:
                key = show_img(img)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if save_folder:
                        os.makedirs(save_folder, exist_ok=True)
                        save_path = os.path.join(save_folder,
                                                 os.path.split(fname)[-1])
                        save_img(save_path, img)
                    index_list.append(index_list.popleft())
                elif key == ord('b'):
                    if current_show >= 0:
                        index_list.appendleft(index_list.pop())
                        current_show -= 2
                    else:
                        print("We now at the start point and can't show previous image.")
                        continue
                else:
                    index_list.append(index_list.popleft())
                current_show += 1
                if current_show >= len(index_list) - 1:
                    print('Exit! All images have been shown.')
                    break
            else:
                print('ignore image, skip!')
                if key == ord('b'):
                    index_list.appendleft(index_list.pop())
                    current_show -= 1
                else:
                    index_list.append(index_list.popleft())
                    current_show += 1
                continue

    def save(self, save_folder=None, specified_imgs=None, **kwargs):
        """Save all data in a specified folder.

        Args:
            save_folder (str): Folder path where the images store.
            specified_imgs: (str, optional): Specified visualization to show, folder or txt file.
        """

        assert save_folder is not None, "save_folder shouldn't be None"
        os.makedirs(save_folder, exist_ok=True)

        index_list = get_index_list(specified_imgs, self.fnames, self.extension)

        for index, fname in enumerate(index_list):
            print(f'idx:{index + 1}, fname:{fname.split("/")[-1]}')
            fname = os.path.join(self.img_prefixs[self.fnames.index(fname)],
                                 fname)
            img, flag = self.img_process(fname, **kwargs)
            if flag:
                save_path = os.path.join(save_folder, os.path.split(fname)[-1])
                save_img(save_path, img)
        print('All images are save!')

    def show_single_image(self, fname, save_folder=None, **kwargs):
        """Show a single image and save it in a specified folder.

        Args:
            fname (str): Path of image.
            save_folder: Folder path where the image may store.
        """
        _, fname = os.path.split(fname)

        fname = os.path.join(self.img_prefixs[self.fnames.index(fname)], fname)
        assert fname in self.img_names, f'{fname} should be in dataset.'

        img, _ = self.img_process(fname, **kwargs)
        show_img(img)

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, os.path.split(fname)[-1])
            save_img(save_path, img)

    @abstractmethod
    def img_process(self, fname, show_ori=False, **kwargs):
        """Image process.

        Args:
            fname (str): Path of image.
            show_ori (bool): Whether show original image. Default: False.
        """

        pass

    def get_single_data(self, fname, category_to_show):
        """Get single image's annotation.

        Args:
            fname (str): Path of image.
            category_to_show (tuple or None): Categories need to be displayed.

        Returns:
            dict: Image's annotation.
        """

        data = {}
        labels = []
        for key, value in self.data.items():
            fname_list = value['img_names'].tolist()
            if fname in fname_list:
                index = fname_list.index(fname)
                anno = {k: v[index] for k, v in value.items() if v is not None}
                if isinstance(anno['labels'], np.ndarray):
                    labels.extend(anno['labels'].tolist())
                else:
                    labels.append(anno['labels'])
            else:
                anno = {}
            data.update({key: anno})
        labels = [self.categories[l] for l in labels]

        if category_to_show is not None and not (set(labels) & set(category_to_show)):
            flag = False
        else:
            flag = True

        return data, labels, flag
