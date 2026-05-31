#%%
import torch
import torch.nn as nn
from torchvision import datasets, transforms as tt
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm

rtx = torch.device('cuda:1')

compose = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(.5,.5,inplace=True)
])

path = '../.data/'
mnist10 = datasets.MNIST(
    root=path,
    train=True,
    download=True,
    transform=compose
)

data_ = torch.utils.data.DataLoader(
    mnist10,
    batch_size=32,
    shuffle=True,
    persistent_workers=True,
    num_workers=8,
    pin_memory=True,
)

#%%
class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.Linear(128,1024),
            nn.Tanh(),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Linear(1024,32*32*1),
            nn.Tanh(),
        )
        for layer in self.seq:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x):
        x = self.seq(x).view(-1,1,32,32)
        return x
    
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.UpsamplingBilinear2d((32,32)),
            nn.Flatten(),
            nn.Linear(32*32,1024),
            nn.Tanh(),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Linear(1024,1),
        )
        for layer in self.seq:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x):
        x = self.seq(x)
        return x
    
gen = Generator().to(rtx)
crit = Critic().to(rtx)
opt = [
    torch.optim.SGD(gen.parameters(),lr=1e-3),
    torch.optim.SGD(crit.parameters(),lr=1e-3)
]
loss_fn = nn.BCEWithLogitsLoss()

#%%
for epoch in range(10):
    dLossMean,gLossMean,c = 0,0,0
    batchEpoch = tqdm(data_,desc=f'Epoch {epoch+1}')
    for img,_ in batchEpoch:
        img = img.to(rtx)

        dis_loss = 0
        noise = torch.randn(img.size(0),128,device=rtx)
        fake_imgs = gen(noise)
        true_logits = crit(img)
        fake_logits = crit(fake_imgs.detach())
        dis_loss += loss_fn(true_logits,torch.ones_like(true_logits,device=rtx))
        dis_loss += loss_fn(fake_logits,torch.zeros_like(fake_logits,device=rtx))
        opt[1].zero_grad()
        dis_loss.backward()
        opt[1].step()

        noise = torch.randn(img.size(0),128,device=rtx)
        fake_imgs = gen(noise)
        fake_logits = crit(fake_imgs)
        gen_loss = loss_fn(fake_logits,torch.ones_like(fake_logits,device=rtx))
        opt[0].zero_grad()
        gen_loss.backward()
        opt[0].step()

        c += 1
        gLossMean += gen_loss.item()
        dLossMean += dis_loss.item()
        batchEpoch.set_postfix({
            'dLoss':f'{dLossMean/c:.4f}',
            'gLoss':f'{gLossMean/c:.4f}'
        })

#%%
k = 5
fig, ax = plt.subplots(k,k,figsize=(8,8))
ax = ax.ravel()
with torch.inference_mode():
    noise = torch.randn(k**2,128,device=rtx)
    img = gen(noise).squeeze().cpu().numpy()

for i in range(k**2):
    ax[i].matshow(img[i])
    ax[i].axis(False)
plt.tight_layout(pad=0)
plt.show()