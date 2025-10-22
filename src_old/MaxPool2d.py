import torch
import torch.nn as nn
import torch.nn.functional as F



# Custom MaxPool2D Class






class MaxPool2D(nn.Module):

    def __init__(self,kernel_size, stride = None, padding = 0 , dilation=1):
        super().__init__()

        self.kernel = torch.ones((kernel_size,kernel_size))
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation


    def forward(self,x):

        B,C,H,W = x.shape
        k = self.kernel

        H_out= W_out = ((H+2*self.padding-(self.dilation*(self.kernel_size-1))-1)//self.stride)+1    # calculate output size

        x = x if self.padding == 0 else self.add_padding(x,self.padding)     #adding padding

        out = torch.zeros((B,C,H_out,W_out))    #output


        d = self.dilation
        s = self.stride

        for b in range(B):

            for c in range(C):

                im = x[b,c]

                for i in range(H_out):
                    for j in range(W_out):

                        out[b,c,i,j] = (im[i:i+s*d:d, j:j+s*d:d] * k).max()

        return out



    def add_padding(self,x,padding):
        B,C,H,W = x.shape
        p = padding

        padded_x = torch.zeros((B,C,H+p*2,W+p*2))
        padded_x[:,:,p:p+H,p:p+W] = x

        return padded_x
