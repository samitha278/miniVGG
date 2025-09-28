import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'


from vgg_v2 import minVGG,Config




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



max_iter = 1000
lr = 3e-4



cnn = minVGG(Config())
cnn = cnn.to(device)
#cnn = torch.compile(cnn)



losses = torch.zeros((max_iter,))

use_fused = torch.cuda.is_available()
optimizer = torch.optim.AdamW(cnn.parameters(),lr = lr ,fused = use_fused)

train_iter = iter(train_data)

for i in range(max_iter):

    t0 = time.time()

    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_data)  # reset
        xb, yb = next(train_iter)

    xb,yb = xb.to(device) , yb.to(device)

    logits , loss = cnn(xb,yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()
    dt = (t1-t0) * 1000 # ms 

    losses[i] = loss.item()

    if i%100 ==0 : print(f'{i}/{max_iter}   {loss.item():.4f}   {dt:.4f} ms')





# --------------------------------------------------------------------------

# Validation accuracy

correct, total = 0, 0
for xb, yb in val_data:
    logits = cnn(xb)
    preds = torch.argmax(logits, dim=-1)
    correct += (preds == yb).sum().item()
    total += yb.size(0)

val_acc = correct / total
print(f"Validation accuracy: {val_acc:.4f}")
