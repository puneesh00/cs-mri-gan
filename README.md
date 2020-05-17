<p align="center">
This is the official implementation code for 
</p>
<h3 align="center">
  <a href="https://arxiv.org/abs/1910.06067">Structure Preserving Compressive Sensing MRI Reconstruction using Generative Adversarial Networks</a>
</h3>
<p align="center">
by 
</p>
<h3 align="center">
  <a href="https://scholar.google.com/citations?user=cn1wdTUAAAAJ&hl=en">Puneesh Deora</a>^&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=ZCSsIokAAAAJ&hl=en">Bhavya Vasudeva</a>^&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=8pffuA4AAAAJ&hl=en">Saumik Bhattacharya</a>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=_eIpqasAAAAJ&hl=en">Pyari Mohan Pradhan</a>&nbsp;&nbsp;
</h3>
<p align="center">
(^ equal contribution), accepted in 
</p>
<h3 align="center">
The IEEE CVPR Workshop on <a href="https://data.vision.ee.ethz.ch/cvl/ntire20/">New Trends in Image Restoration and Enhancement (NTIRE)</a> 2020
</h3>

# Pre-requisites
The code was written with Python 3.6.8 with the following dependencies:
* cuda release 9.0, V9.0.176
* tensorflow 1.12.0
* keras 2.2.4
* numpy 1.16.4
* scikit-image 0.15.0
* matplotlib 3.1.0
* nibabel 2.4.1

This code has been tested in Ubuntu 16.04.6 LTS with 4 NVIDIA GeForce GTX 1080 Ti GPUs (each with 11 GB RAM).

# How to Use
### Preparing data
1. The MICCAI 2013 grand challenge dataset can be downloaded from this [webpage](https://my.vanderbilt.edu/masi/workshops/). It is required to fill a google form and register be able to download the data.
2. Download and save the training and testing data in training-training and training-testing folders, respectively, into the repository folder.
3. Run 'python dataset_load.py' to create the GT dataset.
4. Run 'python usamp_data.py' to create the undersampled dataset. 
5. The 'masks' folder contains the undersampling masks used in this work. The path for the mask can be modified in the aformentioned file, as required.

### Training
1. Run 'python training_model.py' to train the model, after checking the names of paths.

### Testing
#### Testing the trained model:
1. Run 'python test_model.py' to test the model, after checking the names of paths.
#### Testing the pre-trained model:
1. The pre-trained weights are available at: [20% undersampling](https://drive.google.com/open?id=1ygzSDA4V09qVhThiYJ606ec912BYjBfP), [30% undersampling](https://drive.google.com/open?id=1j2PPdPT4nOgW8QmhgHJDjMarJUp6gvM6). Download the required weights in the repository folder.
2. Run 'python test_model.py', after changing the names of paths.

# Citation
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
