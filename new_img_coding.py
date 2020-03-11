import torch
import torch.nn as nn
import numpy as np
from factorized_entropy_model import Entropy_bottleneck
from gaussian_entropy_model import Distribution_for_entropy
from fast_context_model import Context4
from basic_module import ResBlock,Non_local_Block,ResGDN
import cv2 

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        return x.round()
    @staticmethod
    def backward(ctx, g):

        return g


class Trunk(nn.Module):
    def __init__(self,N_channel,M1_block,M2_block,FLAG_non_local,inv=False):
        super(Trunk,self).__init__()
        self.N = int(N_channel)
        self.M1 = int(M1_block)
        self.M2 = int(M2_block)
        self.FLAG1 = bool(FLAG_non_local)
        self.trunk = nn.Sequential()
        self.inv = bool(inv)
        for i in range(self.M1):
            self.trunk.add_module('res1'+str(i),ResBlock(self.N,self.N,3,1,1))
        self.trunk.add_module('resgdn',ResGDN(self.N,self.N,3,1,1,self.inv))
        self.rab = nn.Sequential()
        for i in range(self.M2):
            self.rab.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
        self.rab.add_module('conv1',nn.Conv2d(self.N,self.N,1,1,0))

        self.nrab = nn.Sequential()
        self.nrab.add_module('non_local', Non_local_Block(self.N, self.N // 2))
        for i in range(self.M2):
            self.nrab.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
        self.nrab.add_module('conv1', nn.Conv2d(self.N, self.N, 1, 1, 0))
    def forward(self, x):
        if self.FLAG1==True:
            return self.trunk(x)*torch.sigmoid(self.nrab(x))+x
        else:
            return self.trunk(x)*torch.sigmoid(self.rab(x))+x

class Trunk2(nn.Module):
    def __init__(self,N_channel,M1_block,M2_block,FLAG_non_local):
        super(Trunk2,self).__init__()
        self.N = int(N_channel)
        self.M1 = int(M1_block)
        self.M2 = int(M2_block)
        self.FLAG1 = bool(FLAG_non_local)
        self.trunk = nn.Sequential()
        for i in range(self.M1):
            self.trunk.add_module('res1'+str(i),ResBlock(self.N,self.N,3,1,1))

        self.rab = nn.Sequential()
        for i in range(self.M2):
            self.rab.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
        self.rab.add_module('conv1',nn.Conv2d(self.N,self.N,1,1,0))

        self.nrab = nn.Sequential()
        self.nrab.add_module('non_local', Non_local_Block(self.N, self.N // 2))
        for i in range(self.M2):
            self.nrab.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
        self.nrab.add_module('conv1', nn.Conv2d(self.N, self.N, 1, 1, 0))
    def forward(self, x):
        if self.FLAG1==True:
            return self.trunk(x)*torch.sigmoid(self.nrab(x))+x
        else:
            return self.trunk(x)*torch.sigmoid(self.rab(x))+x




class Enc(nn.Module):
    def __init__(self,num_features,M1,M,N2,N1):
        super(Enc,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.N1 = int(N1)

        # main encoder
        self.conv1 = nn.Sequential(nn.Conv2d(self.n_features,self.M1,5,1,2))
        self.trunk1 = Trunk(self.M1,2,2,False,False)
        self.down1 = nn.Conv2d(self.M1,2*self.M1,5,2,2)
        self.trunk2 = Trunk(2*self.M1,3,4,False,False)
        self.down2 = nn.Conv2d(2 * self.M1, self.M, 5, 2, 2)
        self.trunk3 = Trunk(self.M, 3, 4,False,False)
        self.down3 = nn.Conv2d(self.M, self.M, 5, 2, 2)
        self.trunk4 = Trunk(self.M, 3, 4,False,False)
        self.down4 = nn.Conv2d(self.M, self.N1, 5, 2, 2)
        self.trunk5 = Trunk(self.N1, 3, 4,True,False)

        # hyper encoder
        self.trunk6 = Trunk2(self.N1,2,2,True)
        self.down6 = nn.Conv2d(self.N1,self.M,5,2,2)
        self.trunk7 = Trunk2(self.M,2,2,True)
        self.down7 = nn.Conv2d(self.M,self.M,5,2,2)
        self.conv2 = nn.Conv2d(self.M, self.N2, 3, 1, 1)
        self.trunk8 = Trunk2(self.N2,2,2,True)

    def forward(self, x, middle_map):
        
        x1 = self.conv1(x)
        x1 = self.down1(self.trunk1(x1))
        x2 = self.down2(self.trunk2(x1))
        x3 = self.down3(self.trunk3(x2))
        x4 = self.down4(self.trunk4(x3))
        x5 = self.trunk5(x4)
        middle_map = middle_map.unsqueeze(1)
        B, C, H, W = x5.size()
        B1, C1, H1, W1 = middle_map.size()
        middle_map = middle_map.expand(B1, C, H1, W1)
        x5 = x5*middle_map      
        
        x6 = self.down6(self.trunk6(x5.detach()))
        x7 = self.down7(self.trunk7(x6))
        x8 = self.trunk8(self.conv2(x7))
        
        return [x5,x8]


class Hyper_Dec(nn.Module):
    def __init__(self, N2,M,N1):
        super(Hyper_Dec, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        self.N1 = int(N1)
        # hyper decoder
        self.trunk8 = Trunk2(self.N2, 2, 2, True)
        self.conv2 = nn.Conv2d(self.N2, self.M, 3, 1, 1)
        self.up7 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2, 1)
        self.trunk7 = Trunk2(self.M, 2, 2, True)
        self.up6 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2, 1)
        self.trunk6 = Trunk2(self.M, 2, 2, True)
        self.conv3 = nn.Conv2d(self.M,2*self.N1,3,1,1)


    def forward(self,xq2):
        x7 = self.conv2(self.trunk8(xq2))
        x6 = self.trunk7(self.up6(x7))
        x5 = self.trunk6(self.up7(x6))
        x5 = self.conv3(x5)
        return x5


class Dec(nn.Module):
    def __init__(self,num_features,M1,M,N1):
        super(Dec,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N1 = int(N1)



    # main decoder
        self.trunk5 = Trunk(self.N1, 3, 4, True,True)
        self.up4 = nn.ConvTranspose2d(self.N1, self.M, 5, 2, 2,1)
        self.trunk4 = Trunk(self.M, 3, 4, False,True)
        self.up3 = nn.ConvTranspose2d(self.M, self.M, 5, 2, 2,1)
        self.trunk3 = Trunk(self.M, 3, 4, False,True)
        self.up2 = nn.ConvTranspose2d(self.M, 2*self.M1, 5, 2, 2,1)
        self.trunk2 = Trunk(2 * self.M1, 3, 4, False,True)
        self.up1 = nn.ConvTranspose2d(2*self.M1, self.M1, 5, 2, 2,1)
        self.trunk1 = Trunk(self.M1, 2, 2, False,True)
        self.conv1 = nn.Sequential(nn.Conv2d(self.M1, self.n_features,  5, 1, 2))

    def forward(self,xq1):

        x5 = self.up4(self.trunk5(xq1))
        x4 = self.up3(self.trunk4(x5))
        x3 = self.up2(self.trunk3(x4))
        x2 = self.up1(self.trunk2(x3))
        x1 = self.trunk1(x2)
        x = self.conv1(x1)

        return x

class Intra_coding(nn.Module):
    def __init__(self,num_features,M1,M,N2,N1):
        super(Intra_coding,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.N1 = int(N1)
        self.encoder = Enc(num_features, self.M1, self.M, self.N2,self.N1)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(self.N2, self.M,self.N1)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.decoder = Dec(num_features, self.M1,self.M,self.N1)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, middle_map, is_training):
        x1,x2 = self.encoder(x, middle_map)
        if is_training:
            #xq1 = self.add_noise(x1)     In extreme low bitrate scenario,we found add_noise func in training code
            #xq2 = self.add_noise(x2)     would cause color distortion in compressed image.
            xq1 = RoundNoGradient.apply(x1)
            xq2 = RoundNoGradient.apply(x2)
        else:
            xq1 = torch.round(x1)
            xq2 = torch.round(x2)
        xp2 = self.factorized_entropy_func(xq2)
        hyper_dec = self.hyper_dec(xq2)

        middle_map = middle_map.unsqueeze(1)
        B, C, H, W = xq1.size()
        B1, C1, H1, W1 = middle_map.size()
        middle_map = middle_map.expand(B1, C, H1, W1)

        return [xq1*middle_map, xp2, hyper_dec]


class object_coding(nn.Module):
    def __init__(self,num_features,M1,M,N2,N1):
        super(object_coding,self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.N1 = int(N1)
        self.intra1 =  Intra_coding(3,32,192,42,64)
        self.intra2 =  Intra_coding(3,32,192,42,64)
        self.context_model1 = Context4()
        self.context_model2 = Context4()
        
        self.decoder = Dec(num_features, self.M1,self.M,self.N1)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, img, middle_map, is_training):
        
        bk_xq1, bk_hyper_prob, bk_rec_hyper = self.intra1(img, 1-middle_map, is_training)
        obj_xq1, obj_hyper_prob, obj_rec_hyper = self.intra2(img, middle_map, is_training)
        
        bk_main_prob = self.context_model1(bk_xq1, bk_rec_hyper)
        obj_main_prob = self.context_model2(obj_xq1, obj_rec_hyper)
        
        img_xq1 = bk_xq1 + obj_xq1

        rec_img = self.decoder(img_xq1)
        rec_img = torch.clamp(rec_img,min=0,max=1)

        return [rec_img, (bk_main_prob, bk_hyper_prob), (obj_main_prob, obj_hyper_prob)]