import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_module import ResBlock
from gaussian_entropy_model import Gaussian_entropy_model

class MaskConv3d(nn.Conv3d):
    def __init__(self, mask_type,in_ch, out_ch, kernel_size, stride, padding):
        super(MaskConv3d, self).__init__(in_ch, out_ch, kernel_size, stride, padding,bias=True)

        self.mask_type = mask_type
        ch_out, ch_in, k, k, k = self.weight.size()
        mask = torch.zeros(ch_out, ch_in, k, k, k)
        central_id = k*k*(k//2)+k*(k//2)
        current_id = 1
        if mask_type=='A':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id <= central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1


        self.register_buffer('mask', mask)
    def forward(self, x):

        self.weight.data *= self.mask
        return super(MaskConv3d,self).forward(x)



class Context4(nn.Module):
    def __init__(self):
        super(Context4, self).__init__()
        self.conv1 = MaskConv3d('A', 1, 24, 5, 1, 2)

        self.conv2 = nn.Sequential(nn.Conv3d(25,64,1,1,0),nn.LeakyReLU(),nn.Conv3d(64,96,1,1,0),nn.LeakyReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        self.conv3 = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.LeakyReLU())

        self.gaussin_entropy_func = Gaussian_entropy_model()


    def forward(self, x,hyper):

        pad_w = x.size()[2] % 4
        pad_h = x.size()[3] % 4
        if pad_w != 0:
            pad_w = 4 - pad_w
        if pad_h != 0:
            pad_h = 4 - pad_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top 
        pad_func = nn.ReplicationPad2d((pad_top, pad_bottom, pad_left, pad_right))

        x = pad_func(x)
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper),dim=1)
        x1 = self.conv1(x)

        output = self.conv2(torch.cat((x1,hyper),dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)

        return p3

class Context_model2(nn.Module):
    def __init__(self):
        super(Context_model2, self).__init__()
        self.conv1 = MaskConv3d('A', 1, 24, 5, 1, 2)

        self.conv2 = nn.Sequential(nn.Conv3d(26,64,1,1,0),nn.LeakyReLU(),nn.Conv3d(64,96,1,1,0),nn.LeakyReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        self.conv3 = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.LeakyReLU())

        self.gaussin_entropy_func = Gaussian_entropy_model()


    def forward(self, x,hyper, t_prior):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper),dim=1)
        t_prior = torch.unsqueeze(t_prior,dim=1)
        x1 = self.conv1(x)

        output = self.conv2(torch.cat((x1, hyper,t_prior),dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)

        return p3
