import numpy as np
import os
import os.path
from PIL import Image


def pil_loader(path):
    # print(f"PIL LOADER : Trying to load: {path}")
    if not os.path.exists(path):  
        # print(f"ATTENTION : file  not found: {path}")
        return None 
    with open(path, 'rb') as f:
        img = Image.open(f)
        # if img:
        #     print(f"{img} a bien été chargé")
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        # print("DECOUPAGE :", [x.split(' ')[0] for x in f.readlines()])
        image_index = [x.split(' ')[0].replace('real/', '').replace('clipart/', '') for x in f.readlines()]
        # print("MAKEDATASET FROM LIST : ", image_index)
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    # print("MAKEDATASET FROM LIST : ", image_index)
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        # 
        # ("IMAGELIST image_list:", image_list)
        imgs, labels = make_dataset_fromlist(image_list)
        # print("IMAGELIST  imgs:", imgs)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        # print(f"IMAGELIST_VISDA : path {path}")
        # print(f"IMAGELIST_VISDA : root {self.root}")
        # print(f"IMAGELIST_VISDA : self.imgs[index] {self.imgs[index]}")
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
