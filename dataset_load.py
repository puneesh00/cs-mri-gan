import os
import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
import pickle

save_path='/home/cs-mri-gan/'

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

def load_b(path):
    f = os.listdir(path)
    data = []
    #use imgs with more than 10% non-zero values
    n_zero_ratio = 0.1
    for i in range(len(f)):
      img = os.path.join(path, f[i])
      data_new=np.load(img, allow_pickle =True )
      data_new=data_new.astype('float32')
      for j in range(data_new.shape[0]): 
        if (float(np.count_nonzero(data_new[j,:,:]))/np.prod(data_new[j,:,:].shape))>=n_zero_ratio:
          data_new[j,:,:] = data_new[j,:,:]/127.5-1.0    
          data.append(data_new[j,:,:])
    data = np.asarray(data)
    return data

def train_data_aug(train_gt):
    #for mrnet, replace the indices with the commented ones
    gt1=train_gt[0:13461,:,:]       #0:8100

    gt2a=train_gt[12471:12966,:,:]  #7500:7800    #overlapping data
    gt2b=train_gt[12966:13461,:,:]  #7800:8100    #overlapping data

    gt3a=train_gt[13461:16629,:,:]  #8100:10000   #non-overlapping data
    gt3b=train_gt[16629:19797,:,:]  #10000:11900  #non-overlapping data

    gt2=np.vstack((gt2a,gt2b))
    gt3=np.vstack((gt3a,gt3b))
    gt4 = np.vstack((gt2,gt3))

    gt_new=np.vstack((gt1,gt4))

    return gt_new

#for training data

#miccai dataset
train_path='/home/cs-mri-gan/training-training/warped-images'
train_gt=load_a(train_path,1090)

#mrnet dataset
#train_path='/home/cs-mri-gan/train/coronal'
#train_gt=load_b(train_path)

train_gt_aug=train_data_aug(train_gt) #created gt for augmented data

with open(os.path.join(save_path,'training_gt_aug.pickle'),'wb') as f:
        pickle.dump(train_gt_aug,f,protocol=4)

'''
#for testing data

#miccai dataset
test_path='/home/cs-mri-gan/training-testing/warped-images'
test_data=load_a(test_path, 390)

#mrnet dataset
#test_path='/home/cs-mri-gan/valid/coronal'
#test_data=load_b(test_path)

with open(os.path.join(save_path,'testing_gt.pickle'),'wb') as f:
       pickle.dump(test_data,f,protocol=4)
'''
