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
    batch_size=128,
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
            nn.Linear(256,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024,32*32*1),
            # nn.BatchNorm1d(32*32),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(10,128)
        for layer in self.seq:
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight,.2)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.seq[-2].weight)
        nn.init.zeros_(self.seq[-2].bias)

    def forward(self,x,number):
        x = torch.cat([x,self.embed(number)],-1)
        x = self.seq(x).view(-1,1,32,32)
        return x
    
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.UpsamplingBilinear2d((32,32)),
            nn.Flatten(),
            nn.Linear(32*32*2,1024),
            nn.Tanh(),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Linear(1024,1),
        )
        self.embbed = nn.Embedding(10,32*32*1)
        for layer in self.seq:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x,label):
        x = self.seq[0:2](x)
        embedding = self.embbed(label)
        x = torch.cat([x,embedding],-1)
        x = self.seq[2:](x)
        return x
    
gen = Generator().to(rtx)
crit = Critic().to(rtx)
opt = [
    torch.optim.SGD(gen.parameters(),lr=1e-3/2),
    torch.optim.SGD(crit.parameters(),lr=1e-3/2)
]
gLoss,dLoss = [],[]
loss_fn = nn.BCEWithLogitsLoss()

#%%
for epoch in range(100):
    dLossMean,gLossMean,c = 0,0,0
    batchEpoch = tqdm(data_,desc=f'Epoch {epoch+1}')
    for img,label in batchEpoch:
        img,label = img.to(rtx), label.to(rtx)

        crit.train(); gen.eval()
        dis_loss = 0
        noise = torch.randn(img.size(0),128,device=rtx)
        numbers = torch.randint(0,10,[noise.size(0),],device=rtx)
        fake_imgs = gen(noise,numbers)
        true_logits = crit(img,label)
        fake_logits = crit(fake_imgs.detach(),numbers)
        dis_loss += loss_fn(true_logits,torch.ones_like(true_logits,device=rtx))
        dis_loss += loss_fn(fake_logits,torch.zeros_like(fake_logits,device=rtx))
        opt[1].zero_grad()
        dis_loss.backward()
        opt[1].step()

        crit.eval(); gen.train()
        noise = torch.randn(img.size(0),128,device=rtx)
        numbers = torch.randint(0,10,[noise.size(0)],device=rtx)
        fake_imgs = gen(noise,numbers)
        fake_logits = crit(fake_imgs,numbers)
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
    gLoss.append(gLossMean/c); dLoss.append(dLossMean/c)

#%%
plt.semilogy(gLoss,label='gLoss')
plt.semilogy(dLoss,label='dLoss')
plt.legend()
plt.grid(which='both')
plt.tight_layout()
plt.show()

#%%
k = 5
gen.eval()
with torch.inference_mode():
    noise = torch.randn(k**2,128,device=rtx)
    labels = torch.randint(0,10,[k**2],device=rtx)
    img = gen(noise,labels).squeeze().cpu().numpy()

plt.figure(figsize=(8,8))
for i in range(k**2):
    plt.subplot(k,k,i+1)
    plt.imshow(img[i],cmap='magma')
    plt.title(labels[i].cpu().numpy(),fontsize=12)
    plt.axis(False)
plt.tight_layout(pad=0)
plt.show()