# cs-mri-gan
This is the official implementation code for "[Structure Preserving Compressive Sensing MRI Reconstruction using Generative Adversarial Networks](https://arxiv.org/abs/1910.06067)", by [Puneesh Deora](https://scholar.google.com/citations?user=cn1wdTUAAAAJ&hl=en)^, [Bhavya Vasudeva](https://scholar.google.com/citations?user=ZCSsIokAAAAJ&hl=en)^, [Saumik Bhattacharya](https://scholar.google.com/citations?user=8pffuA4AAAAJ&hl=en), [Pyari Mohan Pradhan](https://scholar.google.com/citations?user=_eIpqasAAAAJ&hl=en), accepted in IEEE CVPR Workshop on NTIRE 2020 (^ equal contribution).

#Pre-requisites
The code was written with Python 3.6.8 with the following dependencies:
cuda release 9.0, V9.0.176
tensorflow 1.12.0
keras 2.2.4
numpy 1.16.4
scikit-image 0.15.0
matplotlib 3.1.0
nibabel 2.4.1
This code has been tested in Ubuntu 16.04.6 LTS with 4 NVIDIA GeForce GTX 1080 Ti GPUs (each with 11 GB RAM).

#How to Use
###Preparing data
The MICCAI 2013 grand challenge dataset can be downloaded from this [webpage](https://my.vanderbilt.edu/masi/workshops/). It is required to fill a google form and register be able to download the data.
Download and save the training and testing data in training-training and training-testing folders, respectively, into the repository folder.
Run 'python dataset_load.py' to create the GT training dataset.
Run 'python usamp_data.py' to create the undersampled training dataset. 
The 'masks' folder contains the undersampling masks used in this work. The path for the mask can be modified in the aformentioned file, as required.

###Training
Run 'python train_model.py' to train the model.

###Testing


#Citation

```
@misc{deora2019structure,
    title={Structure Preserving Compressive Sensing MRI Reconstruction using Generative Adversarial Networks},
    author={Puneesh Deora and Bhavya Vasudeva and Saumik Bhattacharya and Pyari Mohan Pradhan},
    year={2019},
    eprint={1910.06067},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
