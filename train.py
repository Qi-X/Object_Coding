import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import cv2
from dataset import VOCDataset_seg
from torch.utils.data import DataLoader
from new_img_coding import object_coding
import torch_msssim
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device_ids = [0,1,2,3]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_path = './checkpoint'

d = 8
a1 = 2
a2 = 0.5



def train(epochs, learning_rate, batch_size, pre_train):
    compression_model = object_coding(3, 32, 192, 42, 64)
    l2_loss_func = nn.MSELoss()
    loss_func = torch_msssim.MS_SSIM(max_val=1)

    if torch.cuda.is_available():
        print("GPU Used----------")
        compression_model = compression_model.cuda()
        compression_model = torch.nn.DataParallel(compression_model, device_ids = device_ids)
        l2_loss_func = l2_loss_func.cuda()
        loss_func = loss_func.cuda()

    if pre_train:
        print("Restoring the model----------")
        compression_model.module.load_state_dict(torch.load('/data/xq/ckpt_obj/1 0.0203_0.2915.pkl'))


    optimizer = optim.Adam(compression_model.parameters(), lr = learning_rate)

    train_data = VOCDataset_seg('/data/VOCdevkit', [('2007', 'train'), ('2012', 'train')])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) 
        
    print("Begin Training---------")
    
    compression_model = compression_model.train()
    for epoch in range(epochs):
        sum_psnr = 0.0
        sum_msssim = 0.0
        rec_loss = 0.0
        sum_bpp = 0.0
        bk_sum_bpp = 0.0
        obj_sum_bpp = 0.0
        step = 0
        for n, sets in enumerate(train_loader):
            imgs = sets[0]
            seg_maps = sets[1]
            middle_maps = sets[2]
            
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                seg_maps = seg_maps.cuda()
                middle_maps = middle_maps.cuda()
            mse = 0.0
            loss = 0.0
            bk_bpp = 0.0
            obj_bpp = 0.0
            avg_bpp = 0.0
            
            rec_img, bk_prob, obj_prob = compression_model(imgs, middle_maps, is_training=True)
            
            seg_num_pixels =  torch.nonzero(seg_maps[:, :, :]).size()[0]
            mse = l2_loss_func(rec_img[:, :, :, :].clone(), imgs[:, :, :, :].clone())
            loss = 1 - loss_func(rec_img[:, :, :, :].clone(), imgs[:, :, :, :].clone())
            num_pixels = imgs.size()[0] * imgs.size()[2] * imgs.size()[3]
            
            bk_bpp = cal_bpp(*bk_prob,  num_pixels-seg_num_pixels)
            obj_bpp = cal_bpp(*obj_prob, seg_num_pixels)
            avg_bpp = cal_avg_bpp(*bk_prob, *obj_prob, num_pixels)
        

            if bk_bpp.item() > 5:
                bk_bpp = 0.0

            sum_loss = d *loss + a1*bk_bpp + a2*obj_bpp 
            optimizer.zero_grad()
            sum_loss.backward()
            optimizer.step()

            sum_psnr = sum_psnr + psnr(mse.item())
            sum_msssim = sum_msssim + (-10*np.log10(loss.item()))
            rec_loss = rec_loss + loss.item()
            bk_sum_bpp = bk_sum_bpp + bk_bpp.item()
            obj_sum_bpp = obj_sum_bpp + obj_bpp.item()
            sum_bpp = sum_bpp + avg_bpp.item()

            
            if (step+1) % 100 == 0:
                print('psnr:', sum_psnr/(step+1), 'MS-SSIM', sum_msssim/(step+1), 
                        'bk_bpp:', bk_sum_bpp/(step+1), 'obj_bpp:', obj_sum_bpp/(step+1), 'sum_bpp:', sum_bpp/(step+1))
                

            if (step+1) % 500 == 0:
                torch.save(compression_model.module.state_dict(), './checkpoint/%d %.4f_%.4f.pkl' % (epoch, rec_loss/(step+1), sum_bpp/(step+1)))
            
            step += 1

        sum_psnr = sum_psnr / step
        sum_msssim = sum_msssim / step
        rec_loss = rec_loss / step
        bk_sum_bpp = bk_sum_bpp / step
        obj_sum_bpp = obj_sum_bpp / step
        sum_bpp = sum_bpp / step
        print("epoch[{}]: psnr:{:.6f} ms-ssim:{:.6f} bk_avg_bpp:{:.6f} obj_avg_bpp:{:.6f} sum_avg_bpp:{:.6f}".format(
                        epoch, sum_psnr, sum_msssim, bk_sum_bpp, obj_sum_bpp, sum_bpp))
        torch.save(compression_model.module.state_dict(), '/data/xq/ckpt_obj/%d %.4f_%.4f.pkl' % (epoch, rec_loss, sum_bpp))



def cal_bpp(main_prob, hyper_prob, num_pixels):
    return torch.sum(torch.log(main_prob))/(-np.log(2) * num_pixels) + torch.sum(torch.log(hyper_prob))/(-np.log(2)* num_pixels)

def cal_avg_bpp(bk_main_prob, bk_hyper_prob, obj_main_prob, obj_hyper_prob, num_pixels):
    return (torch.sum(torch.log(bk_main_prob))/(-np.log(2)) + torch.sum(torch.log(bk_hyper_prob))/(-np.log(2)) \
            + torch.sum(torch.log(obj_main_prob))/(-np.log(2)) + torch.sum(torch.log(obj_hyper_prob))/(-np.log(2)))/num_pixels

def psnr(mse):
	return 20. * np.log10(1.) - 10. * np.log10(mse)


if __name__ == '__main__':
	train(100, 0.00001, 4, True)