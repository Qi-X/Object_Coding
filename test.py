import torch
import numpy as np
import cv2
import math 
import torch_msssim
from new_img_coding import object_coding
from dataset import VOCDataset_seg
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import res_models.resnet_dilated as resnet_dilated
from sklearn.metrics import confusion_matrix


txt_path = "E:\\dataset\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\val.txt"
ids = []
for line in open(txt_path):
    ids.append(line.strip())

segpath = "E:\\dataset\\VOCdevkit\\VOC2012\\SegmentationClass\\%s.png"
imgpath = "E:\\dataset\\VOCdevkit\\VOC2012\\JPEGImages\\%s.jpg"

def get_mask(index):
    img_id = ids[index]

    transform = transforms.Compose(
                [
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
    img_not_preprocessed = Image.open(imgpath % img_id).convert('RGB')
    img = transform(img_not_preprocessed)
    img = img.unsqueeze(0).cuda()

    seg_model = resnet_dilated.Resnet34_8s(num_classes=21)
    seg_model.load_state_dict(torch.load('resnet_34_8s_68.pth'))
    seg_model.cuda()
    seg_model.eval()

    res = seg_model(img)
    _, tmp = res.squeeze(0).max(0)
    seg = tmp.data.cpu().numpy().squeeze().astype(np.float32)
    #cv2.imshow('mask', seg)
    #cv2.waitKey(0)
    H_ORG, W_ORG = seg.shape
    H_PAD = int(64.0 * np.ceil(H_ORG / 64.0))
    W_PAD = int(64.0 * np.ceil(W_ORG / 64.0))
    seg_pad = np.zeros([H_PAD, W_PAD], dtype='float32')
    seg_pad[:H_ORG,:W_ORG] = seg

    seg_num_pixels = len(np.nonzero(seg_pad)[0])
    seg_pad = cv2.resize(seg_pad, (int(np.ceil(W_ORG / 64.0))*4, int(np.ceil(H_ORG / 64.0))*4))
    seg_pad = ((seg_pad != 0) + 0).astype(np.float32)

    return torch.from_numpy(seg_pad), seg_num_pixels

def get_img(index):
    img_id = ids[index]
    
    img = cv2.imread(imgpath % img_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32).transpose(2, 0, 1)/255.0
    C, H_ORG, W_ORG = img.shape
    H_PAD = int(64.0 * np.ceil(H_ORG / 64.0))
    W_PAD = int(64.0 * np.ceil(W_ORG / 64.0))
    im = np.zeros([C, H_PAD, W_PAD], dtype='float32')
    im[:, :H_ORG,:W_ORG] = img

    return torch.from_numpy(im), H_ORG, W_ORG 



def load_network(checkpoint_path):
    compression_model = object_coding(3, 32, 192, 42, 64).cuda()
    compression_model.load_state_dict(torch.load(checkpoint_path))

    return compression_model

def psnr(img1, img2):
    mse = np.mean( np.square(img1 - img2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse)

def cal_bpp(main_prob, hyper_prob, num_pixels):
    return torch.sum(torch.log(main_prob))/(-np.log(2) * num_pixels) + torch.sum(torch.log(hyper_prob))/(-np.log(2)* num_pixels)

def cal_avg_bpp(bk_main_prob, bk_hyper_prob, obj_main_prob, obj_hyper_prob, num_pixels):
    return (torch.sum(torch.log(bk_main_prob))/(-np.log(2)) + torch.sum(torch.log(bk_hyper_prob))/(-np.log(2)) \
            + torch.sum(torch.log(obj_main_prob))/(-np.log(2)) + torch.sum(torch.log(obj_hyper_prob))/(-np.log(2)))/num_pixels

def test(checkpoint_path):
    compression_model = load_network(checkpoint_path)
    compression_model.eval()

    msssim_func = torch_msssim.MS_SSIM(max_val=1.0).cuda()
    PSNR = 0.0
    ms_ssim = 0.0 
    bk_avg_bpp = 0.0
    obj_avg_bpp = 0.0
    sum_avg_bpp = 0.0
    
    num_examples = 0

    for i in range(len(ids)):
        
        img, H_ORG, W_ORG = get_img(i)
        middle_mask, seg_num_pixels = get_mask(i)

        img = img.cuda()
        middle_mask = middle_mask.cuda()
        img = torch.unsqueeze(img, 0)

        with torch.no_grad():
            rec_img, bk_prob, obj_prob = compression_model(img, middle_mask, is_training=False)
            rec_img = torch.clamp(rec_img, min=0.0, max=1.0)
            rec_img = rec_img[:, :, :H_ORG, :W_ORG]
                        
            psnr_ = psnr(rec_img[:, :, :H_ORG, :W_ORG].cpu().numpy()*255.0, img[:, :, :H_ORG, :W_ORG].cpu().numpy()*255.0)
            PSNR += psnr_
            msssim = msssim_func(rec_img[:, :, :H_ORG, :W_ORG], img[:, :, :H_ORG, :W_ORG]).item()
            ms_ssim += msssim
            bk = cal_bpp(*bk_prob, H_ORG*W_ORG-seg_num_pixels).item()
            if bk > 10.0:
                bk = 0.0
            bk_avg_bpp += bk
            obj = cal_bpp(*obj_prob, seg_num_pixels).item()
            if obj > 10.0:  # if seg_num_pixels = 0
                obj = 0.0
            obj_avg_bpp += obj
            sum_ = cal_avg_bpp(*bk_prob, *obj_prob, H_ORG*W_ORG).item()
            sum_avg_bpp += sum_
            
        num_examples += 1
        #break

    PSNR /= num_examples
    ms_ssim /= num_examples
    bk_avg_bpp /= num_examples
    obj_avg_bpp /= num_examples
    sum_avg_bpp /= num_examples

    print("PSNR:", PSNR)
    print("ms_ssim:", ms_ssim, -10*np.log10(1-np.mean(ms_ssim)))
    print("bk_avg_bpp:", bk_avg_bpp)
    print("obj_avg_bpp:", obj_avg_bpp)
    print("sum_avg_bpp:", sum_avg_bpp)



if __name__ == '__main__':
	test("./checkpoint/0 0.090739_0.060983.pkl")
