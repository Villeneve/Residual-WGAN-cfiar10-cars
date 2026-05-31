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
    tt.Resize((37,37)),
    tt.RandomCrop((32,32)),
    tt.ToTensor(),
    tt.Normalize(.5,.5)
])

path = '../../.data/'
cifar10 = datasets.CIFAR10(
    root=path,
    train=True,
    download=True,
    transform=compose
)
mask = np.where(np.array(cifar10.targets)==1)[0]
cifar10 = torch.utils.data.Subset(cifar10,mask)

data_ = torch.utils.data.DataLoader(
    cifar10,
    batch_size=512,
    shuffle=True,
    persistent_workers=True,
    num_workers=8,
    pin_memory=True,
)

#%%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class ResConv2D(nn.Module):
    def __init__(self, inCh, outCh, spectral=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not spectral:
            self.path = nn.Sequential(
                nn.InstanceNorm2d(inCh,affine=True),
                nn.LeakyReLU(.1),
                nn.Conv2d(inCh,outCh,3,1,1,padding_mode='reflect'),
                nn.InstanceNorm2d(outCh,affine=True),
                nn.LeakyReLU(.1),
                nn.Conv2d(outCh,outCh,3,1,1,padding_mode='reflect'),
            )
        else:
            self.path = nn.Sequential(
                nn.InstanceNorm2d(inCh,affine=True),
                nn.LeakyReLU(.1),
                nn.utils.spectral_norm(nn.Conv2d(inCh,outCh,3,1,1,padding_mode='reflect')),
                nn.InstanceNorm2d(outCh,affine=True),
                nn.LeakyReLU(.1),
                nn.utils.spectral_norm(nn.Conv2d(outCh,outCh,3,1,1,padding_mode='reflect')),
            )

        self.skip = nn.Identity() if inCh == outCh else nn.Conv2d(inCh,outCh,1,1,0)

    def forward(self,x):
        skip = self.skip(x)
        x = self.path(x)
        return (x+skip)/2**.5
    
class SelfAttention2D(nn.Module):
    def __init__(self, inCh, spectral=False, div_by=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not spectral:
            self.Q = nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0)
            self.K = nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0)
            self.V = nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0)
        else:
            self.Q = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
            self.K = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
            self.V = nn.utils.spectral_norm(nn.Conv2d(inCh,inCh,1,1,0))
        self.gamma = nn.Parameter(torch.zeros((1,inCh,1,1)))

    def forward(self,x):
        wQ = self.Q(x)
        wK = self.K(x)
        wV = self.V(x)
        B,C,W,H = wQ.size()
        attention = wQ.view(B,C,-1).transpose(-1,-2)@wK.view(B,C,-1)
        attention /= C**.5
        attention = nn.functional.softmax(attention,-1)
        output = attention@wV.view(B,wV.size(1),-1).transpose(-1,-2)
        output = output.transpose(-1,-2).view(B,-1,W,H)
        return x + self.gamma*output
    
class MultiHeadAttention2D(nn.Module):
    def __init__(self, inCh, n_heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_head = nn.ModuleList([SelfAttention2D(inCh,div_by=n_heads) for i in range(n_heads)])

    def forward(self,x):
        head_list = [self.multi_head[i](x) for i in range(len(self.multi_head))]
        return torch.cat(head_list,1)

#%%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.Linear(128,4*4*512),
            nn.Unflatten(-1,(512,4,4)),
            nn.InstanceNorm2d(512,affine=True),
            nn.LeakyReLU(.1),

            nn.UpsamplingBilinear2d(scale_factor=2),
            ResConv2D(512,256),
            ResConv2D(256,256),

            nn.UpsamplingBilinear2d(scale_factor=2),
            ResConv2D(256,128),
            ResConv2D(128,128),

            nn.UpsamplingBilinear2d(scale_factor=2),
            ResConv2D(128,64),
            ResConv2D(64,64),
            MultiHeadAttention2D(64,8),

            nn.Conv2d(64,3,1,1,0),
            nn.Tanh()
        )
        for layer in self.seq:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x):
        x = self.seq(x)
        return x
 
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.UpsamplingBilinear2d((32,32)),

            ResConv2D(3,32,spectral=True),
            nn.AvgPool2d(2,2), 

            SelfAttention2D(32,spectral=True),

            ResConv2D(32,64,spectral=True),
            nn.AvgPool2d(2,2),

            ResConv2D(64,128,spectral=True),
            nn.AvgPool2d(2,2),
            

            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(4*4*128,1)),
        )

    def forward(self,x:torch.Tensor):
        return self.seq(x)
    
gen = Generator().to(rtx)
crit = Critic().to(rtx)
opt = [
    torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(.0,.99),),
    torch.optim.AdamW(crit.parameters(),lr=1e-4,betas=(.0,.99),weight_decay=1e-3)
]
gLoss,dLoss = [],[]
loss_fn = nn.BCEWithLogitsLoss()

#%%––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
for epoch in range(500):
    dLossMean,gLossMean,c = 0,0,0
    batchEpoch = tqdm(data_,desc=f'Epoch {epoch+1}')
    for img,_ in batchEpoch:
        img = img.to(rtx)

        crit.train(); gen.eval()
        dis_loss = 0
        noise = torch.randn(img.size(0),128,device=rtx)
        fake_imgs = gen(noise)
        true_logits = crit(img)
        fake_logits = crit(fake_imgs.detach())
        # dis_loss += loss_fn(true_logits,torch.ones_like(true_logits,device=rtx))
        # dis_loss += loss_fn(fake_logits,torch.zeros_like(fake_logits,device=rtx))
        dis_loss += (true_logits-1).square().mean()
        dis_loss += (fake_logits-0).square().mean()
        # dis_loss += -true_logits.mean() + fake_logits.mean()
        # dis_loss += 1e-3*true_logits.square().mean()
        opt[1].zero_grad()
        dis_loss.backward()
        opt[1].step()

        crit.eval(); gen.train()
        noise = torch.randn(img.size(0),128,device=rtx)
        fake_imgs = gen(noise)
        fake_logits = crit(fake_imgs)
        # gen_loss = loss_fn(fake_logits,torch.ones_like(fake_logits,device=rtx))
        gen_loss = (fake_logits-1).square().mean()
        # gen_loss = -fake_logits.mean()
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
    plt.plot(gLoss,label='gLoss')
    plt.plot(dLoss,label='dLoss')
    plt.axhline((gLoss[-1]+dLoss[-1])/2,color='r',linestyle='--',label='mean')
    plt.legend()
    plt.grid(which='both')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('loss.png')
    plt.clf()
    plt.close()

    if epoch%1 == 0:
        k = 5
        gen.eval()
        with torch.inference_mode():
            noise = torch.randn(k**2,128,device=rtx)
            img = gen(noise).permute(0,2,3,1).cpu().numpy()*127.5+127.5
            img = img.astype(np.uint8)

        plt.figure(figsize=(8,8))
        for i in range(k**2):
            plt.subplot(k,k,i+1)
            plt.imshow(img[i],cmap='magma')
            plt.axis(False)
        plt.tight_layout(pad=0)
        plt.savefig('img.png')
        plt.close()

#%%––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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