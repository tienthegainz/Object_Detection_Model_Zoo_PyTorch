import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    "balo", "balo_diutre", "bantreem", "binh_sua", "cautruot", "coc_sua", "ghe_an",
    "ghe_bap_benh", "ghe_ngoi_oto", "ghedualung_treem", "ke", "noi", "person", "phao",
    "quay_cui", "tham", "thanh_chan_cau_thang", "thanh_chan_giuong", "xe_babanh", "xe_choichan",
    "xe_day", "xe_tapdi", "xichdu", "yem", )


def get_list_ids(path):
    """
        path: image_path or xml_path
        output: list of [image, xml] item name
    """
    ids = list()
    for item in os.listdir(path):
        parts = item.split('.')[:-1]
        ids.append('.'.join(parts))
    return ids


class VOCDataset(data.Dataset):
    """VOC Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 mode='train',
                 transform=None,
                 dataset_name='FTECH_prj_imageq'):
        self.root = root
        self.transform = transform
        self.name = dataset_name
        if mode == 'train':
            self._annopath = osp.join(self.root, 'train_xml', '%s.xml')
            self._imgpath = osp.join(self.root, 'train_image', '%s')
            self.ids = get_list_ids(osp.join(self.root, 'train_image'))
        elif mode == 'val':
            self._annopath = osp.join(self.root, 'val_xml', '%s.xml')
            self._imgpath = osp.join(self.root, 'val_image', '%s')
            self.ids = get_list_ids(osp.join(self.root, 'val_image'))
        else:
            print('%s not supported. Exitting\n' % mode)
            exit(-1)

    def __getitem__(self, index):
        img_id = self.ids[index]
        name = self._annopath % img_id
        target = ET.parse(name).getroot()
        image_name = img_id+'.'+target.find('./filename').text.split('.')[-1]
        img = cv2.imread(self._imgpath % image_name)
        try:
            if len(img.shape) > 2 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
        except Exception as err:
            print(err, '---', image_name)
        height, width, channels = img.shape

        target = np.array(target)
        sample = {'img': img, 'annot': target}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(VOC_CLASSES)

    def label_to_name(self, label):
        return VOC_CLASSES[label]

    def load_annotations(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        gt = np.array(gt)
        return gt
