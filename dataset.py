import os 
import torch 
import torch.utils.data as data 
import cv2
import numpy as np 
import xml.etree.ElementTree as ET 

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

class VOCDataset_seg(data.Dataset):
    '''
    Arguments:
    rootpath(string):filepath to VOCdevkit folder '../../VOCdevkit'
    image_sets(string):imageset to use(eg. [('2007', 'trainval'), ('2012', 'trainval')])
    '''
    def __init__(self, rootpath, image_sets):

        self.rootpath = rootpath
        self.image_sets = image_sets
        self._segpath = os.path.join('%s', 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()

        for (year, name) in image_sets:
            filepath = os.path.join(self.rootpath, 'VOC'+year)
            for line in open(os.path.join(filepath, 'ImageSets', 'Segmentation', name+'.txt')):
                self.ids.append((filepath, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]

        seg_img = cv2.imread(self._segpath % img_id)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2GRAY)
        seg_img = cv2.resize(seg_img, (320, 320))
        middle_map = cv2.resize(seg_img, (20, 20)) 
        seg_map = ((seg_img != 0) + 0).astype(np.float32)
        middle_map = ((middle_map != 0) + 0).astype(np.float32)
        
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320)).astype(np.float32).transpose(2, 0, 1)/255.0

        return torch.from_numpy(img), torch.from_numpy(seg_map), torch.from_numpy(middle_map)

    def __len__(self):
        return len(self.ids)


