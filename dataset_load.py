import os
import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
import pickle

save_path='/home/diencephalon/'

def load_a(path, num):
        f = os.listdir(path)
        a = len(f)
        data = []
        #use imgs with more than 10% non-zero values
        n_zero_ratio = 0.1
        #num is to reduce the number of files loaded
        for i in range(len(f)-num):
            img = os.path.join(path, f[i])
            img_l = nib.load(img)
            img_data = img_l.get_fdata()
            vol_max = np.max(img_data)
            img_data = img_data/vol_max*2
            for j in range(img_data.shape[2]): 
                if (float(np.count_nonzero(img_data[:,:,j]))/np.prod(img_data[:,:,j].shape))>=n_zero_ratio:
                    img_data[:,:,j] = img_data[:,:,j]-1   
                    img_data_ts = np.rot90(img_data[:,:,j])
                    data.append(img_data_ts)
        data = np.asarray(data)
        return data

def train_data_aug(train_gt):

    gt1=train_gt[0:13461,:,:]

    gt2a=train_gt[12471:12966,:,:]#overlapping data
    gt2b=train_gt[12966:13461,:,:]#overlapping data

    gt3a=train_gt[13461:16629,:,:]#non-overlapping data
    gt3b=train_gt[16629:19797,:,:]#non-overlapping data

    gt2=np.vstack((gt2a,gt2b))
    gt3=np.vstack((gt3a,gt3b))
    gt4 = np.vstack((gt2,gt3))

    gt_new=np.vstack((gt1,gt4))

    return gt_new

#for training data
train_path='/home/diencephalon/training-training/warped-images'
train_gt=load_a(train_path,1130)

train_gt_aug=train_data_aug(train_gt) #created gt for augmented data

with open(os.path.join(save_path,'training_gt_aug.pickle'),'wb') as f:
        pickle.dump(train_gt_aug,f,protocol=4)

'''
#for testing data
test_path='/home/diencephalon/training-testing/warped-images'
test_data=load_a(test_path, 5)
with open(os.path.join(save_path,'testing_gt.pickle'),'wb') as f:
       pickle.dump(test_data,f,protocol=4)
'''