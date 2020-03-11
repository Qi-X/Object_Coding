import numpy as np 
import torch 

def psnr(mse):
	return 20. * np.log10(1.) - 10. * np.log10(mse)

def mse(img1, img2):
	return np.mean(np.square(img1 - img2))

def psnr2(mse):
	return 20. * np.log10(255.) - 10. * np.log10(mse)

def cal_bpp(main_prob, hyper_prob, num_pixels):
    return torch.sum(torch.log(main_prob))/(-np.log(2)* num_pixels)+torch.sum(torch.log(hyper_prob))/(-np.log(2)* num_pixels)