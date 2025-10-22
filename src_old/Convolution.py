import torch
import torch.nn as nn
import torch.nn.functional as F



# Custom Convolution 2D Class


class Conv2d(nn.Module):

  def __init__(self,in_channels, out_channels, kernel_size,stride = 1, padding = 0, dilation= 1,bias = True):
    super().__init__()

    assert kernel_size%2==1 , "kernel size must be odd"

    self.weights = torch.randn((out_channels,in_channels,kernel_size,kernel_size)) * (in_channels**-0.5)
    self.bias = torch.randn(out_channels) if bias else None



    #variables
    self.out_channels = out_channels
    self.in_channels = in_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation





  def forward(self,x):

    B,C,H,W = x.shape

    H_out= W_out = ((H+2*self.padding-(self.dilation*(self.kernel_size-1))-1)//self.stride)+1    # calculate output size

    x = x if self.padding == 0 else self.add_padding(x,self.padding)     #adding padding

    feature_map = torch.zeros((B,self.out_channels,H_out,W_out))    #output feature map

    for b in range(B):
      for out_c in range(self.out_channels):

        feature_map[b,out_c,:,:] = torch.add(*[self.Convolution(x[b,i],self.weights[out_c,i]) for i in range(self.in_channels)])
        if self.bias is not None:
          feature_map[b,out_c,:,:] += self.bias[out_c]





    return feature_map





  def Convolution(self,x, kernel):

    a, b = kernel.shape
    s = self.stride
    d1 = self.dilation

    feature_map = []
    r, c = x.shape

    for i in range(0, r - a*d1 + 1, s):
        row = []
        for j in range(0, c - b*d1 + 1, s):

            temp = (x[i:i+a*d1:d1, j:j+b*d1:d1] * kernel).sum(dim=(0,1))
            row.append(temp)
        feature_map.append(row)

    return torch.tensor(feature_map)






  def add_padding(self,x,padding):
    B,C,H,W = x.shape
    p = padding

    padded_x = torch.zeros((B,C,H+p*2,W+p*2))
    padded_x[:,:,p:p+H,p:p+W] = x

    return padded_x



