import torch
import torch.nn as nn
import torch.nn.functional as F


import time
import math

from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vgg_v3 import minVGG,Config

from data.dataset import get_dataloaders


tt1 = time.time()



torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)


# --------------------------------------------------------------------------

batch_size = 32

train_data, val_data = get_dataloaders("/home/samitha/Projects/datasets/imagenet100",batch_size = batch_size)

# --------------------------------------------------------------------------



# Initialize Model
config = Config()
cnn = minVGG(config)
cnn = cnn.to(device)
cnn_compiled = torch.compile(cnn)


# ------------------------------------------------------------------------------
# Learning Rate Schedule

max_iter = 200000 # 1000
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


# Log Dir ----------------------------------------------------------------------

log_dir = "log_SGD_wM"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

with open(log_file, "w") as f: 
    pass 


# ------------------------------------------------------------------------------

use_fused = torch.cuda.is_available()

# optimizer with weight decay
optimizer = torch.optim.AdamW(cnn.parameters(),lr = max_lr ,fused = use_fused,weight_decay=0.1)

# optimizer = torch.optim.SGD(cnn.parameters(),lr=max_lr,momentum=0.9,weight_decay=0.1, fused = use_fused)

final_step = max_iter-1
train_iter = iter(train_data)


# Trin loop --------------------------------------------------------

for i in range(max_iter):
    
    
    # Validation ---------------------------------------------------
    if i % 500 == 0 or i==final_step:
        cnn_compiled.eval()
        with torch.no_grad():
            
            val_loss = 0.0
            val_step = 0
            for x,y in val_data:
                x,y = x.to(device),y.to(device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16):
                    logits,loss = cnn_compiled(x,y)
                
                val_loss += loss.detach()
                val_step+=1
            val_loss /= val_step
        
        print(f'val loss : {val_loss.item():.4f}')
        with open(log_file, "a") as f:
            f.write(f"{i} val {val_loss.item():.4f}\n")
                
        cnn_compiled.train()
        
    
    # Save Checkpoint ---------------------------------------------
    
    if i>0 and (i%20000 == 0 or i == final_step):
        checkpoint_path = os.path.join(log_dir, f"model_{i:05d}.pt")
        checkpoint = {
            'model': cnn.state_dict(),
            'config': config,
            'step': i,
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
    # --------------------------------------------------------------
        
    

    t0 = time.time()
    try:
        xb , yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_data)
        xb , yb = next(train_iter)
        
    xb,yb = xb.to(device) , yb.to(device)
    
    with torch.autocast(device_type=device , dtype=torch.bfloat16):
        logits , loss = cnn_compiled(xb,yb)

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

    print(f'{i}  {loss.item():.4f}  {dt:.4f} ms   norm:{norm.item():.4f}    lr:{lr:.4e}')
    
    with open(log_file, "a") as f:
        f.write(f"{i} train {loss.item():.6f}\n")



tt2 = time.time()

dtt = (tt2 - tt1) * 1000 # ms

time_file = os.path.join(log_dir, f"time_total.txt")

with open(time_file, "w") as f: 
    f.write(f"total training time : {dtt}\n")
 
