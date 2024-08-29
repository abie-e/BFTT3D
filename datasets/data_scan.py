"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_scanobjectnn_data(split, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    if split == 1:
        h5_name = BASE_DIR + '/../data/h5_files/main_split/' + partition + '_objectdataset.h5'
    elif split == 2:
        h5_name = BASE_DIR + '/../data/h5_files/main_split_nobg/' + partition + '_objectdataset.h5'
    elif split == 3:
        h5_name = BASE_DIR + '/../data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'

    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, num_points, split=2, partition='training'):
        self.data, self.label = load_scanobjectnn_data(split, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def load_data_c(data_path,corruption,severity):

    if corruption == 'original':
        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '.npy')
    else:
        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label
    
class ScanObjectNNC(Dataset):
    def __init__(self, split='test', test_data_path="./data/scanobjectnn_c/", train_data_path="./data/scanobjectnn_c/", corruption="uniform", severity=5):
        self.corruption = corruption
        self.severity = severity
        self.data_path = {
            "test":  test_data_path,
            "train": train_data_path
        }[split]
        self.data, self.label = load_data_c(self.data_path, self.corruption, self.severity)


    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
