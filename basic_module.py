import torch.nn as nn
import torch
from GDN_transform import GDN
from torch.autograd import Variable
import torch.nn.functional as f


class ResGDN(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding,inv=False):
        super(ResGDN,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.inv = bool(inv)
        self.conv1 = nn.Conv2d(self.in_ch,self.out_ch,self.k, self.stride
                                             ,self.padding)
        self.conv2 = nn.Conv2d(self.in_ch,self.out_ch,self.k, self.stride
                                             ,self.padding)
        self.ac1 = GDN(self.in_ch,self.inv)
        self.ac2 = GDN(self.in_ch,self.inv)


    def forward(self,x):
        x1 = self.ac1(self.conv1(x))
        x2 = self.conv2(x1)
        out = self.ac2(x + x2)
        return out


class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
        super(ResBlock,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride
                               , self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride
                               , self.padding)

    def forward(self,x):
        x1 = self.conv2(f.relu(self.conv1(x)))
        out = x+x1
        return out


# here use embedded gaussian
class Non_local_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Non_local_Block,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel,self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self,x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size,self.out_channel,-1)
        g_x = g_x.permute(0,2,1)
        theta_x = self.theta(x).view(batch_size,self.out_channel,-1)
        theta_x = theta_x.permute(0,2,1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x,phi_x)
        f_div_C = f.softmax(f1,dim=-1)
        y = torch.matmul(f_div_C,g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size,self.out_channel,*x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z

class ConvLSTMCell(nn.Module):
    def __init__(self,input_dim,hidden_dim,kernel_size,stride):
        super(ConvLSTMCell,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - stride)//2
        self.conv = nn.Conv2d(self.input_dim+self.hidden_dim,4 * hidden_dim,self.kernel_size,self.stride,self.padding)

    def forward(self, input_tensor, h_cur=None,c_cur=None):

        if h_cur is None:
            h_cur = self.init_hidden(input_tensor.size(0),input_tensor.size(2),input_tensor.size(3))
        if c_cur is None:
            c_cur = self.init_hidden(input_tensor.size(0), input_tensor.size(2), input_tensor.size(3))

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = f.sigmoid(cc_i)
        f_ = f.sigmoid(cc_f)
        o = f.sigmoid(cc_o)
        g = f.tanh(cc_g)

        c_next = f_ * c_cur + i * g
        h_next = o * f.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size,H,W):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, H, W)).cuda())


class Tran_Bloc(nn.Module):
    def __init__(self,in_channel,M1_block,M2_block,FLAG_NLB=True):
        super(Tran_Bloc,self).__init__()
        self.N = int(in_channel)
        self.M1 = int(M1_block)
        self.M2 = int(M2_block)
        self.FLAG = bool(FLAG_NLB)

        if self.FLAG:
            self.main_branch = nn.Sequential()
            for i in range(self.M1):
                self.main_branch.add_module('res1'+str(i),ResBlock(self.N,self.N,3,1,1))

            self.atten_branch = nn.Sequential()
            for i in range(self.M2):
                self.atten_branch.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
            self.atten_branch.add_module('conv1',nn.Conv2d(self.N,self.N,1,1,0))

        else:
            self.main_branch = nn.Sequential()
            for i in range(self.M1):
                self.main_branch.add_module('res1'+str(i),ResBlock(self.N,self.N,3,1,1))

            self.atten_branch = nn.Sequential()
            self.atten_branch.add_module('non_local', Non_local_Block(self.N, self.N // 2))
            for i in range(self.M2):
                self.atten_branch.add_module('res2'+str(i),ResBlock(self.N,self.N,3,1,1))
            self.atten_branch.add_module('conv1',nn.Conv2d(self.N,self.N,1,1,0))

    def forward(self, x):
            return self.main_branch(x)*torch.sigmoid(self.atten_branch(x))+x