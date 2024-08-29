import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_shapenet_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR + '/../data'
    all_data = []
    all_label = []
    
    # path_h5py_all = list()
    # path_h5py = os.path.join(BASE_DIR, '%s*.h5' % type)
    # paths = glob.glob(path_h5py)
    paths = glob.glob(os.path.join(DATA_DIR, 'shapenetcorev2_hdf5_2048', '%s*.h5'%partition))

    for h5_name in paths:
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


class ShapeNet(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_shapenet_data(partition)
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def load_data_c(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label
    
class ShapeNetC(Dataset):
    def __init__(self, num_points=1024, split='test', test_data_path="./data/shapenet_c/", train_data_path="./data/shapenet_c/", corruption="uniform", severity=5):
        self.corruption = corruption
        self.severity = severity
        self.data_path = {
            "test":  test_data_path,
            "train": train_data_path
        }[split]
        self.data, self.label = load_data_c(self.data_path, self.corruption, self.severity)
        self.num_points = num_points


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
