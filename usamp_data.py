import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import cmath


def usam_data(accel, data_tensor, mask):
    data =[] 
    for i in range(data_tensor.shape[0]):
    	fourier = np.fft.fft2(data_tensor[i,:,:])
    	cen_fourier  = np.fft.fftshift(fourier)
    	subsam_fourier = np.multiply(cen_fourier,mask) #undersampling in k-space
    	uncen_fourier = np.fft.ifftshift(subsam_fourier)
    	zro_image = np.fft.ifft2(uncen_fourier) #zero-filled reconstruction
    	data.append(zro_image)        
    data = np.asarray(data)
    return data 


def usam_data_noise(accel, data_tensor, mask, noise_ratio):
    fft_data=[]
    data =[]
    for i in range(data_tensor.shape[0]):
        fourier = np.fft.fft2(data_tensor[i,:,:])
        cen_fourier  = np.fft.fftshift(fourier)
        fft_data.append(cen_fourier)
    fft_data=np.asarray(fft_data)
    fft_std=np.std(fft_data)
    nstd=(noise_ratio*fft_std)/np.sqrt(2)
    insh=(fft_data.shape[1],fft_data.shape[2])
    for k in range(fft_data.shape[0]):    
        fft_noise=fft_data[k,:,:]+np.random.normal(0,nstd,insh)+1j*np.random.normal(0,nstd,insh) #adding noise
        subsam_fourier = np.multiply(fft_noise,mask) #undersampling in k-space
        uncen_fourier = np.fft.ifftshift(subsam_fourier)
        zro_image = np.fft.ifft2(uncen_fourier) #zero-filled reconstruction
        data.append(zro_image) 
        
    data = np.asarray(data)
    return data 

save_path='/home/cs-mri-gan/'
mask_path='/home/cs-mri-gan/masks/mask_1dg_a10.pickle' #path for the required mask
maf=open(mask_path,'rb')
mask=pickle.load(maf)

accel=10 #acceleration factor

#creating undersampled training data
train_path='/home/cs-mri-gan/training_gt_aug.pickle'
trf=open(train_path,'rb')
train_data=pickle.load(trf)

train_data_new = usam_data(accel, train_data[0:13461,:,:], mask) #nonoise
train_data_new2 = usam_data_noise(accel, train_data[13461:13956,:,:], mask, 0.1) #10%noise-overlapping
train_data_new3 = usam_data_noise(accel, train_data[13956:14451,:,:], mask, 0.2) #20%noise-overlapping
train_data_new4 = usam_data_noise(accel, train_data[14451:17619,:,:,], mask, 0.1) #10%noise-nonoverlapping
train_data_new5 = usam_data_noise(accel, train_data[17619:20787,:,:], mask, 0.2) #20%noise-nonoverlapping

stack1 = np.vstack((train_data_new,train_data_new2))
stack2 = np.vstack((train_data_new3, train_data_new4))
stack3 = np.vstack((stack2, train_data_new5))
fstack = np.vstack((stack1, stack3))

with open(os.path.join(save_path,'training_usamp_1dg_a10_aug.pickle'),'wb') as f:
	pickle.dump(fstack,f,protocol=4)

'''
#creating undersampled testing data
test_path='/home/cs-mri-gan/testing_gt.pickle'
tef=open(test_path,'rb')
test_data=pickle.load(tef)

train_data_new = usam_data(accel, test_data,mask) #for noise-free imgs
#test_data_new=usam_data_noise(accel,test_data,mask,0.1) #for imgs with 10%noise
#test_data_new=usam_data_noise(accel,test_data,mask,0.2) #for imgs with 20%noise
with open(os.path.join(save_path,'testing_usamp_1dg_a10.pickle'),'wb') as f:
	pickle.dump(test_data_new,f,protocol=4)
'''