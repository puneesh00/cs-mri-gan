This is the official implementation code for **[Structure Preserving Compressive Sensing MRI Reconstruction using Generative Adversarial Networks](https://arxiv.org/abs/1910.06067)** by *[Puneesh Deora](https://scholar.google.com/citations?user=cn1wdTUAAAAJ&hl=en)*^, *[Bhavya Vasudeva](https://scholar.google.com/citations?user=ZCSsIokAAAAJ&hl=en)*^, *[Saumik Bhattacharya](https://scholar.google.com/citations?user=8pffuA4AAAAJ&hl=en)*, *[Pyari Mohan Pradhan](https://scholar.google.com/citations?user=_eIpqasAAAAJ&hl=en)*, accepted in IEEE CVPR Workshop on [New Trends in Image Restoration and Enhancement (NTIRE)](https://data.vision.ee.ethz.ch/cvl/ntire20/) 2020 (^ equal contribution).

## Pre-requisites
The code was written with Python 3.6.8 with the following dependencies:
* cuda release 9.0, V9.0.176
* tensorflow 1.12.0
* keras 2.2.4
* numpy 1.16.4
* scikit-image 0.15.0
* matplotlib 3.1.0
* nibabel 2.4.1
* cuDNN 7.4.1

This code has been tested in Ubuntu 16.04.6 LTS with 4 NVIDIA GeForce GTX 1080 Ti GPUs (each with 11 GB RAM).

## How to Use
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

## Citation
```
@article{deora2019structure,
    title={Structure Preserving Compressive Sensing MRI Reconstruction using Generative Adversarial Networks},
    author={P. Deora and B. Vasudeva and S. Bhattacharya and P. M. Pradhan},
    journal={ArXiv},
    year={2019},
    volume={abs/1910.06067}
}
```

## License
```
   Copyright 2020 Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
