import json
import pickle

import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PolypDataset(CustomDataset):

    CLASSES = ['polyp']

    def load_annotations(self, ann_file):
        return pickle.load(open(ann_file,'rb'))


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []

        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                if len(self.data_infos[i]['ann']['bboxes'])!= 0:
                    valid_inds.append(i)

        return valid_inds