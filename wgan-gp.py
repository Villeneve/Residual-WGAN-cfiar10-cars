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
    tt.Resize((35,35),tt.InterpolationMode.NEAREST),
    tt.RandomCrop((32,32)),
    tt.ColorJitter(.2,.2,.2,.05),
    tt.Pad(20,padding_mode='reflect'),
    tt.RandomRotation(10,tt.InterpolationMode.NEAREST,),
    tt.CenterCrop((32,32)),
    tt.RandomHorizontalFlip(),
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
class RMSNorm2D(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1))

    def forward(self, x:torch.Tensor):
        rms = x.square().mean([1,2,3],keepdim=True) + 1e-8
        rms = rms.sqrt()
        return self.gamma*x/(rms+1e-8)
    
class ResConv2D(nn.Module):
    def __init__(self, inCh, outCh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.path = nn.Sequential(
            nn.Conv2d(inCh,outCh,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outCh,outCh,3,1,1),
        )
        nn.init.kaiming_normal_(self.path[0].weight,0.01)
        nn.init.xavier_normal_(self.path[2].weight)
        nn.init.zeros_(self.path[0].bias)
        nn.init.zeros_(self.path[2].bias)

        if inCh == outCh:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(inCh,outCh,1,1,0)
            nn.init.xavier_normal_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

        self.gamma = nn.Parameter(torch.ones(1,1,1,1)*0.2)
        # self.preNorm = RMSNorm2D(inCh)
        # self.posNorm = RMSNorm2D(outCh)

    def forward(self,x):
        return self.skip(x) + self.gamma * self.path(x)

class SelfAttention2D(nn.Module):
    def __init__(self, inCh, div_by=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.div_by = div_by
        self.Q = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
        self.K = nn.utils.spectral_norm(nn.Conv2d(inCh,max(inCh//div_by,1),1,1,0))
        self.V = nn.utils.spectral_norm(nn.Conv2d(inCh,inCh,1,1,0))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        wQ = self.Q(x)
        wK = self.K(x)
        wV = self.V(x)
        B,C,W,H = wQ.size()
        attention = wQ.view(B,C,-1).transpose(-1,-2)@wK.view(B,C,-1)
        # attention = torch.einsum('ikj,ikl->ijl',wQ.view(B,C,-1),wK.view(B,C,-1))
        attention /= C**.5
        attention = nn.functional.softmax(attention,-1)
        output = attention@wV.view(B,wV.size(1),-1).transpose(-1,-2)
        output = output.transpose(-1,-2).view(B,-1,W,H)

        return x+self.gamma*output

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

class NoiseInject(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1)*.05)

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise

class PixelNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x: torch.Tensor):
        scale = x.square().mean(1,keepdim=True)+1e-8
        return x*scale.rsqrt()

def critic_score(global_score:torch.Tensor, patch_score:torch.Tensor, weight_patch=0.25):
    global_score = global_score.view(global_score.size(0))
    patch_score = patch_score.flatten(1).mean(1)
    return global_score+weight_patch*patch_score
#%%–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mappingNetwork = nn.Sequential(
            nn.Linear(128,1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024,4*4*512),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(-1,(512,4,4)),
        )
        for layer in self.mappingNetwork:
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight,.01)
                nn.init.zeros_(layer.bias)

        self.seq = nn.Sequential(
            self.mappingNetwork,

            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.PixelShuffle(2),
            ResConv2D(512,256),
            ResConv2D(256,128),

            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.PixelShuffle(2),
            NoiseInject(128),
            ResConv2D(128,128),
            ResConv2D(128,128),
            ResConv2D(128,64),

            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.PixelShuffle(2),
            NoiseInject(64),
            ResConv2D(64,64),
            ResConv2D(64,64),
            ResConv2D(64,64),
            ResConv2D(64,64),
            # OptimizedMultiHeadAttention2D(64,8),

            nn.Conv2d(64,3,3,1,1,padding_mode='reflect'),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.seq[-2].weight)
        nn.init.zeros_(self.seq[-2].bias)

    def forward(self,x):
        x = self.seq(x)
        return x
 
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            # nn.UpsamplingBilinear2d((32,32)),
            ResConv2D(4,32),
            ResConv2D(32,32),
            # nn.PixelUnshuffle(2),
            nn.AvgPool2d(2,2),
            # 16x16

            ResConv2D(32,32),
            ResConv2D(32,32),
            ResConv2D(32,32),
            ResConv2D(32,64),
            # nn.PixelUnshuffle(2),
            nn.AvgPool2d(2,2),
            # 8x8

            # OptimizedMultiHeadAttention2D(32,4,spectral=False),
            # SelfAttention2D(32,spectral=False),

            ResConv2D(64,64),
            ResConv2D(64,64),
            ResConv2D(64,64),
            ResConv2D(64,128),
            # nn.PixelUnshuffle(2),
            nn.AvgPool2d(2,2),
            # 4x4

            ResConv2D(128,128),
            ResConv2D(128,128),
            ResConv2D(128,128),
            ResConv2D(128,512),
            nn.Conv2d(512,512,4,1,0),

            nn.Flatten(),
            nn.Linear(512,1)
        )
        self.convpatch = nn.Conv2d(64,1,1,1,0)

    def forward(self,x:torch.Tensor):
        std = x.std(0,keepdim=True).mean([1,2,3],keepdim=True).expand(x.size(0),1,x.size(2),x.size(3))
        x = torch.cat([x,std],1)
        x = self.seq[0:8](x)
        local_patch = self.convpatch(x)
        global_patch = self.seq[8:](x)
        return global_patch, local_patch
    
gen = Generator().to(rtx)
crit = Critic().to(rtx)
opt = [
    torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(0.0,0.9)),
    torch.optim.Adam(crit.parameters(),lr=1e-4,betas=(0.0,0.9)),
]
gLoss,dLoss,hgrads,hwdistance = [],[],[],[]
loss_fn = nn.BCEWithLogitsLoss()
k=5
# noise_state = torch.randn(k**2,128,device=rtx)

#%%––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
counter = 0
for epoch in range(5000):
    dLossMean,gLossMean,c = 0,0,0
    batchEpoch = tqdm(data_,desc=f'Epoch {epoch+1}')
    crit.train(); gen.train(); gen_loss = torch.zeros(1)

    EMA_grads = None
    EMA_wdistance = None
    EMA_dloss = None
    EMA_gloss = None
    beta = 0.95

    for n,(img,_) in enumerate(batchEpoch):

        img = img.to(rtx)
        dis_loss = 0
        noise = torch.randn(img.size(0),128,device=rtx)
        fake_imgs = gen(noise)
        true_logits_global,true_logits_patch = crit(img)
        fake_logits_global,fake_logits_patch = crit(fake_imgs.detach())
        true_score = critic_score(true_logits_global,true_logits_patch)
        fake_score = critic_score(fake_logits_global,fake_logits_patch)
        # WGAN
        alpha = torch.rand((img.size(0),1,1,1),device=rtx)
        interpolate = alpha*fake_imgs.detach() + (1-alpha)*img
        interpolate.requires_grad_(True)
        interpolate_out_global,interpolate_out_patch = crit(interpolate)
        interpolate_out_score = critic_score(interpolate_out_global,interpolate_out_patch)
        grads = torch.autograd.grad(
            interpolate_out_score.sum(),
            interpolate,
            create_graph=True,
        )[0]

        grads = grads.flatten(1).norm(2,-1)
        dis_loss += -true_score.mean() + fake_score.mean()
        gp = 10*(grads-1).square().mean()

        EMA_wdistance = -dis_loss.item() if EMA_wdistance is None else beta*EMA_wdistance + (1-beta)*(-dis_loss.item())
        EMA_grads = grads.mean().item() if EMA_grads is None else beta*EMA_grads + (1-beta)*grads.mean().item()
        dis_loss += gp
        EMA_dloss = dis_loss.item() if EMA_dloss is None else beta*EMA_dloss+(1-beta)*dis_loss.item()
        
        opt[1].zero_grad()
        dis_loss.backward()
        opt[1].step()
        counter += 1
        

        if counter >= 5:
            counter = 0
            noise = torch.randn(img.size(0),128,device=rtx)
            fake_imgs = gen(noise)
            fake_logits_global,fake_logits_patch = crit(fake_imgs)
            fake_logits = critic_score(fake_logits_global,fake_logits_patch)
            # gen_loss = loss_fn(fake_logits,torch.ones_like(fake_logits,device=rtx))
            # gen_loss = (fake_logits-1).square().mean()
            # gen_loss = -fake_logits.mean()
            # Hinge loss
            gen_loss = -fake_logits.mean()
            opt[0].zero_grad()
            gen_loss.backward()
            opt[0].step()
            EMA_gloss = gen_loss.item() if EMA_gloss is None else beta*EMA_gloss+(1-beta)*gen_loss.item()

        if EMA_gloss is not None:
            batchEpoch.set_postfix({
                'dLoss':f'{EMA_dloss:.4f}',
                'gLoss':f'{EMA_gloss:.4f}',
                'grads':f'{EMA_grads:.4f}',
                'wdistance':f'{EMA_wdistance:.4f}'
            })

    gLoss.append(EMA_gloss); dLoss.append(EMA_dloss)
    hgrads.append(EMA_grads); hwdistance.append(EMA_wdistance)
    plt.plot(gLoss,label='gLoss')
    plt.plot(dLoss,label='dLoss')
    plt.plot(hgrads,label='Grads')
    plt.plot(hwdistance,label='wDistance')
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