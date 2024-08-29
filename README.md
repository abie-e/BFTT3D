# BFTT3D
This repository is for implementing frameworks introduced in the following paper:  
**"Backpropagation-free Network for 3D Test-time Adaptation"**

<p align="center">
  <img width="900" src="https://github.com/abie-e/BFTT3D/blob/main/Figure1.png"> 
</p>

### Installation
Create a conda environment and install dependencies:
```
conda env create -f environment.yml
```
If this fails, please try to install the Point-NN and MATE environments one by one.
1.[Point-NN](https://github.com/ZrrSkywalker/Point-NN)
2.[MATE](https://github.com/jmiemirza/MATE)

### Dataset

#### Folder structure
    .
    ├── ...
    ├── datasets
    ├── data                    
    │   ├── modelnet40_ply_hdf5_2048          
    │   ├── modelnet40_c         
    │   └── ...                
    ├── checkpoint                   
    │   └── ... 
    └── ....


Please download the following datasets for the prototype memory

[Download ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

[Download ScanObjectNN](https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip)

[Download ShapeNetCoreV2](https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/) 

#### Corruption:
Please follow the script from [ModelNet40-C](https://github.com/jiachens/ModelNet40-C/tree/master/data) to add corruption.

We also provide download links:

[Download ModelNet40-C from Google Drive.](https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX?usp=sharing)

[Download ScanObjectNN-C from Google Drive.](https://drive.google.com/file/d/1PpS4oPoPA03-RWTiVp-huvhqa7jh3PbW/view?usp=sharing)


#### Checkpoint:
[checkpoint](https://drive.google.com/drive/folders/1tD8cXVCgGwH5Q6N7PKldp3pG2_tBqxMM?usp=sharing) is uploaded here.

## BFTT3D
For ModelNet40 dataset, just run:
```bash
python run_BFTT3D_mn40.py --pth pointnet
```

The code is not fully organized and maybe a bit cumbersome. Refactoring and finalization are underway.


