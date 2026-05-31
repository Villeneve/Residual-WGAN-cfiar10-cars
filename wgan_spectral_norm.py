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
    tt.ColorJitter(.2,.2,.2,.05),
    tt.Pad(20,padding_mode='reflect'),
    tt.RandomRotation(20,tt.InterpolationMode.BILINEAR,),
    tt.CenterCrop((32,32)),
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
    batch_size=64,
    shuffle=True,
    persistent_workers=True,
    num_workers=8,
    pin_memory=True,
)

#%%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class ResConv2D(nn.Module):
    def __init__(self, inCh, outCh, spectral=False, norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not spectral and norm:
            self.path = nn.Sequential(
                nn.InstanceNorm2d(inCh,affine=True),
                nn.LeakyReLU(.1),
                nn.Conv2d(inCh,outCh,3,1,1,padding_mode='reflect'),
                nn.InstanceNorm2d(outCh,affine=True),
                nn.LeakyReLU(.1),
                nn.Conv2d(outCh,outCh,3,1,1,padding_mode='reflect'),
            )
        elif spectral and not norm:
            self.path = nn.Sequential(
                nn.LeakyReLU(.1),
                nn.utils.spectral_norm(nn.Conv2d(inCh,outCh,3,1,1,padding_mode='reflect')),
                nn.LeakyReLU(.1),
                nn.utils.spectral_norm(nn.Conv2d(outCh,outCh,3,1,1,padding_mode='reflect')),
            )
        else:
            self.path = nn.Sequential(
                nn.LeakyReLU(.1),
                nn.Conv2d(inCh,outCh,3,1,1,padding_mode='reflect'),
                nn.LeakyReLU(.1),
                nn.Conv2d(outCh,outCh,3,1,1,padding_mode='reflect'),
            )

        for layer in self.path:
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,.1)
                nn.init.zeros_(layer.bias)

        if inCh == outCh:
            self.skip = nn.Identity()
        elif spectral:
            self.skip = nn.utils.spectral_norm(nn.Conv2d(inCh,outCh,1,1,0))
        else:
            self.skip = nn.Conv2d(inCh,outCh,1,1,0)

    def forward(self,x):
        skip = self.skip(x)
        x = self.path(x)
        return (x+skip)/2**.5

class SelfAttention2D(nn.Module):
    def __init__(self, inCh, spectral=False, div_by=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not spectral:
            self.Q = nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0)
            self.K = nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0)
            self.V = nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0)
        else:
            self.Q = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
            self.K = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
            self.V = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
        self.gamma = nn.Parameter(torch.zeros((1,max(inCh//div_by,1),1,1)))
        self.div_by = div_by

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
        if self.div_by == 1:
            return x+self.gamma*output
        return self.gamma*output

class OptimizedMultiHeadAttention2D(nn.Module):
    def __init__(self, inCh, n_heads=1, spectral=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert inCh % n_heads == 0, "inCh precisa ser divisível por n_heads"
        
        self.n_heads = n_heads
        self.head_dim = inCh // n_heads
        self.scale = self.head_dim ** -0.5

        # conv_layer = nn.utils.spectral_norm(nn.Conv2d(inCh, inCh, 1, 1, 0)) if spectral else nn.Conv2d(inCh, inCh, 1, 1, 0)
        
        self.Q = nn.utils.spectral_norm(nn.Conv2d(inCh, inCh, 1, 1, 0)) if spectral else nn.Conv2d(inCh, inCh, 1, 1, 0)
        self.K = nn.utils.spectral_norm(nn.Conv2d(inCh, inCh, 1, 1, 0)) if spectral else nn.Conv2d(inCh, inCh, 1, 1, 0)
        self.V = nn.utils.spectral_norm(nn.Conv2d(inCh, inCh, 1, 1, 0)) if spectral else nn.Conv2d(inCh, inCh, 1, 1, 0)
        
        self.proj_out = nn.utils.spectral_norm(nn.Conv2d(inCh, inCh, 1, 1, 0)) if spectral else nn.Conv2d(inCh, inCh, 1, 1, 0)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        N = W * H
        q = self.Q(x)  # (B, C, W, H)
        k = self.K(x)  # (B, C, W, H)
        v = self.V(x)  # (B, C, W, H)
        
        q = q.view(B, self.n_heads, self.head_dim, N).transpose(-1, -2)
        k = k.view(B, self.n_heads, self.head_dim, N)
        v = v.view(B, self.n_heads, self.head_dim, N).transpose(-1, -2)
        
        attention = (q @ k) * self.scale
        attention = nn.functional.softmax(attention, dim=-1)
        out = attention @ v
        out = out.transpose(-1, -2).contiguous().view(B, C, W, H)
        out = self.proj_out(out)
        
        return x + self.gamma * out

#%%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mappingNetwork = nn.Sequential(
            nn.Linear(128,1024),
            nn.LeakyReLU(.1),
            nn.Linear(1024,4*4*512),
        )
        for layer in self.mappingNetwork:
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight,.1)
                nn.init.zeros_(layer.bias)

        self.seq = nn.Sequential(
            self.mappingNetwork,
            nn.Unflatten(-1,(512,4,4)),
            # nn.BatchNorm2d(512,affine=True),
            nn.LeakyReLU(.1),

            nn.UpsamplingBilinear2d(scale_factor=2),
            ResConv2D(512,256),
            ResConv2D(256,256),

            nn.UpsamplingBilinear2d(scale_factor=2),
            ResConv2D(256,128),
            ResConv2D(128,128),
            ResConv2D(128,128),

            nn.UpsamplingBilinear2d(scale_factor=2),
            ResConv2D(128,64),
            ResConv2D(64,64),
            ResConv2D(64,64),
            ResConv2D(64,64),
            # OptimizedMultiHeadAttention2D(64,8),

            nn.Conv2d(64,3,1,1,0),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.seq[-2].weight)

    def forward(self,x):
        x = self.seq(x)
        return x
 
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            # nn.UpsamplingBilinear2d((32,32)),
            nn.utils.spectral_norm(nn.Conv2d(4,32,7,1,3)),
            nn.LeakyReLU(.1),

            ResConv2D(32,32,spectral=True,norm=False),
            ResConv2D(32,32,spectral=True,norm=False),
            ResConv2D(32,32,spectral=True,norm=False),
            ResConv2D(32,32,spectral=True,norm=False),
            nn.Conv2d(32,32,2,2,0),

            # OptimizedMultiHeadAttention2D(32,4,spectral=True),
            # SelfAttention2D(32,spectral=False),

            ResConv2D(32,64,spectral=True,norm=False),
            ResConv2D(64,64,spectral=True,norm=False),
            ResConv2D(64,64,spectral=True,norm=False),
            ResConv2D(64,64,spectral=True,norm=False),
            nn.utils.spectral_norm(nn.Conv2d(64,64,2,2,0)),

            ResConv2D(64,128,spectral=True,norm=False),
            ResConv2D(128,128,spectral=True,norm=False),
            ResConv2D(128,128,spectral=True,norm=False),
            ResConv2D(128,256,spectral=True,norm=False),
            nn.utils.spectral_norm(nn.Conv2d(256,256,2,2,0)),

            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(4*4*256,1)),
        )

    def forward(self,x:torch.Tensor):
        std = x.std(0,keepdim=True).mean([1,2,3],keepdim=True).expand(x.size(0),1,x.size(2),x.size(3))
        x = torch.cat([x,std],1)
        return self.seq(x)
    
gen = Generator().to(rtx)
crit = Critic().to(rtx)
opt = [
    torch.optim.RMSprop(gen.parameters(),lr=1e-4,),
    torch.optim.RMSprop(crit.parameters(),lr=1e-4,),
]
gLoss,dLoss = [],[]
loss_fn = nn.BCEWithLogitsLoss()
k=5
# noise_state = torch.randn(k**2,128,device=rtx)

#%%––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
for epoch in range(5000):
    dLossMean,gLossMean,c = 0,0,0
    batchEpoch = tqdm(data_,desc=f'Epoch {epoch+1}')
    crit.train(); gen.train()
    for n,(img,_) in enumerate(batchEpoch):
        img = img.to(rtx)

        dis_loss = 0
        noise = torch.randn(img.size(0),128,device=rtx)
        fake_imgs = gen(noise)
        true_logits = crit(img)
        fake_logits = crit(fake_imgs.detach())
        # DCGAN
        # dis_loss += loss_fn(true_logits,torch.ones_like(true_logits,device=rtx))
        # dis_loss += loss_fn(fake_logits,torch.zeros_like(fake_logits,device=rtx))
        # LSGAN
        # dis_loss += (true_logits-1).square().mean()
        # dis_loss += (fake_logits-0).square().mean()
        # WGAN
        # dis_loss += -true_logits.mean() + fake_logits.mean()# + 1e-4*true_logits.square().mean()
        # Hinge Loss
        # dis_loss += nn.functional.relu(-true_logits+1).mean() + nn.functional.relu(fake_logits+1).mean()
        # soft Loss
        dis_loss += nn.functional.softplus(-true_logits+1).mean() + nn.functional.softplus(fake_logits+1).mean()
        
        opt[1].zero_grad()
        dis_loss.backward()
        opt[1].step()
        

        if n % 5 == 0:
            noise = torch.randn(img.size(0),128,device=rtx)
            fake_imgs = gen(noise)
            fake_logits = crit(fake_imgs)
            # gen_loss = loss_fn(fake_logits,torch.ones_like(fake_logits,device=rtx))
            # gen_loss = (fake_logits-1).square().mean()
            # gen_loss = -fake_logits.mean()
            # Hinge loss
            gen_loss = -fake_logits.mean()
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