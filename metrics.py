from skimage import measure


def metrics(true_tensor, test_tensor,max_val):
    psnrt = 0
    ssimt = 0
    for i in range(true_tensor.shape[0]): 

    	 psnr = measure.compare_psnr(true_tensor[i,:,:], test_tensor[i,:,:],data_range=max_val)
    	 ssim = measure.compare_ssim(true_tensor[i,:,:], test_tensor[i,:,:],data_range=max_val)
    	 psnrt = psnrt+psnr
    	 ssimt = ssimt+ssim

    psnrt = psnrt/true_tensor.shape[0]
    ssimt = ssimt/true_tensor.shape[0]
    return psnrt, ssimt
