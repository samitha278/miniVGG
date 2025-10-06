import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

import time
import math

from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'


from vgg_v3 import minVGG,Config

print(device)


# --------------------------------------------------------------------------

batch_size = 64


transform_train = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform_train)


train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# --------------------------------------------------------------------------



# Train Model





torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)




cnn = minVGG(Config())
cnn = cnn.to(device)
cnn = torch.compile(cnn)





# ------------------------------------------------------------------------------
# Learning Rate Schedule

max_iter = 1000 # 1000
max_lr = 3e-4
min_lr = max_lr * 0.01
warm_up = max_iter * 0.05


def get_lr(i):

    if i < warm_up :
        return (max_lr/warm_up) * (i+1)

    if i > max_iter : 
        return min_lr

    # cosine decay
    diff = max_lr - min_lr
    steps = max_iter - warm_up
    lr = (diff/2) * math.cos(i * (math.pi / steps)) + diff
    return lr


# ------------------------------------------------------------------------------


losses = torch.zeros((max_iter,))
lrs = torch.zeros((max_iter,))
norms = torch.zeros((max_iter,))

use_fused = torch.cuda.is_available()

# optimizer with weight decay
optimizer = torch.optim.AdamW(cnn.parameters(),lr = max_lr ,fused = use_fused,weight_decay=0.1)

train_iter = iter(train_data)

for i in range(max_iter):

    t0 = time.time()
    try:
        xb , yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_data)
        xb , yb = next(train_iter)
        
    xb,yb = xb.to(device) , yb.to(device)
    logits , loss = cnn(xb,yb)

    optimizer.zero_grad()
    loss.backward()

    # learning rate schedule
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Inplace gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(cnn.parameters(),1.0)

    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()
    dt = (t1-t0) * 1000 # ms 

    losses[i] = loss.item()
    lrs[i] = lr
    norms[i] = norm.item()

    if i%100 ==0 : print(f'{i}/{max_iter}  {loss.item():.4f}  {dt:.4f} ms   norm:{norm.item():.4f}    lr:{lr:.4e}')
