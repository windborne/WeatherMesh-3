import torch
import torch.nn as nn
import torch.nn.functional as F
from model_latlon.vars import get_constant_vars
from utils import * 
import matepoint


POLE_PAD_MODE = 'reflect'

def earth_pad2d(x, pad):
    """ 
    :param x: (N, C, H, W) input tensor
    :param pad: (H_pad, W_pad) tuple of padding
    :return: (N, C, H + 2 * H_pad, W + 2 * W_pad) padded tensor
    """
    H,W = x.shape[2], x.shape[3]
    #assert (H-1) / W == 0.5, f"(H-1)/W ratio must be 0.5, this is an earth grid. Got ({H} - 1)/{W}. x.shape={x.shape}"

    H_pad, W_pad = pad
    
    x = F.pad(x, (0, 0, H_pad, H_pad), mode=POLE_PAD_MODE)

    x = F.pad(x, (W_pad, W_pad, 0, 0), mode='circular') 
    return x

def earth_wrap2d(x,target_W,Wpad):
    # Does the horizonal wrapping for a tensor after it has gone through a conv transpose to go up
    
    B,C,H,W  = x.shape
    assert W > target_W, "The input x width must be larger than the target output with, as the input should have post-convtranspose padding"
    RWpad = W - target_W - Wpad
    if Wpad == 1: assert RWpad == 0
    elif Wpad > 1: assert RWpad > 0, "Really cannot have too many asserts"

    xout = x[:,:,:,Wpad:target_W+Wpad] #select just the part we need
    xout[:,:,:,-Wpad:] += x[:,:,:,:Wpad] #wrap left side to right side
    if RWpad > 0:
        xout[:,:,:,:RWpad] += x[:,:,:,-RWpad:] #wrap right side to left side 
    return xout

def southpole_pad2d(x):
    H,W = x.shape[2], x.shape[3]
    assert H % 2 == 0

    x = F.pad(x, (0, 0, 0, 1), mode=POLE_PAD_MODE)
    return x 

def southpole_unpad2d(x):
    H,W = x.shape[2], x.shape[3]
    assert H % 2 == 1
    x = x[:,:,:-1,:]

    return x 



class EarthResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate_channels=None, time_emb_dim=None, dropout=0.0, kernel_size=3, affine=True, use_pole_convs=False):
        super(EarthResBlock2d, self).__init__()
        self.kernel_size = kernel_size
        self.use_pole_convs = use_pole_convs
        
        if intermediate_channels is None:
            intermediate_channels = out_channels

        self.norm1 = nn.InstanceNorm2d(in_channels, affine=affine)
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=kernel_size, padding=0)
        if use_pole_convs: self.pole_conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, padding=0)
        self.norm2 = nn.InstanceNorm2d(intermediate_channels, affine=affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=kernel_size, padding=0)
        if use_pole_convs: self.pole_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t=None):
        k = self.kernel_size // 2
        h = self.norm1(x)
        h = F.silu(h)
        h = earth_pad2d(h, (k,k))
        h = self.conv1(h)
        if self.use_pole_convs:
            h[:,:,0:1,:] = self.pole_conv1(h[:,:,0:1,:])
            h[:,:,-1:,:] = self.pole_conv1(h[:,:,-1:,:])

        if t is not None:
            time_emb = self.time_mlp(t)
            time_emb = time_emb[:, :, None, None]  # (N, C, 1, 1)
            h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = earth_pad2d(h, (k,k))
        h = self.conv2(h)
        if self.use_pole_convs:
            h[:,:,0:1,:] = self.pole_conv2(h[:,:,0:1,:])
            h[:,:,-1:,:] = self.pole_conv2(h[:,:,-1:,:])

        return h + self.nin_shortcut(x)
    
class EarthConvDown2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(EarthConvDown2d, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        k = self.kernel_size // 2
        h = earth_pad2d(x, (k,k))

        h = self.conv(h)
        return h

class EarthConvUp2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(EarthConvUp2d, self).__init__()
        assert kernel_size % 2 == 1
        assert stride == 2
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,bias=False)

    def forward(self, x):
        B,C,Hs,Ws = x.shape
        H,W = Hs*self.stride-(Hs % 2), Ws*self.stride
        pad = self.kernel_size // 2
        h = self.conv(x)
        h = earth_wrap2d(h,W,pad)
        h_unpad = h[:,:,pad:H+pad,:] 
        assert h_unpad.shape[2:] == (H,W), h_unpad.shape
        return h_unpad


def imsave2(path,t):
    print("-----")
    print(f"{path}: {[x for x in t.shape]}")
    plt.imsave(path,t[0,0].detach().numpy())
    print("----")

def find_periodicity(tensor):
    if tensor.dim() != 4:
        raise ValueError("Input tensor must be 4-dimensional (B,C,H,W)")
    
    B, C, H, W = tensor.shape
    
    def find_period(vec):
        for period in range(1, W // 2 + 1):
            if W % period == 0:
                if torch.all(torch.eq(vec.view(-1, period), vec[:period])):
                    return period
        return W  # If no periodicity found, return the full length
    
    # Create a tensor to store the periodicities
    periodicities = torch.zeros((B, C, H), dtype=torch.int64, device=tensor.device)
    
    for b in range(B):
        for c in range(C):
            for h in range(H):
                periodicities[b, c, h] = find_period(tensor[b, c, h])
    
    return periodicities[0,0].numpy()


def call_checkpointed(module, *args, checkpoint_type="matepoint_sync", **kwargs):
    if not torch.is_grad_enabled():
        checkpoint_type = "none"
    if checkpoint_type == "matepoint": checkpoint_type = "matepoint_sync"
    try: n = module.__name__
    except: n = module.__class__.__name__
   
    if checkpoint_type == "none": return module(*args, **kwargs) # so that it's easy to disable checkpointing for debugging 
    elif checkpoint_type == "torch": return torch.utils.checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False, preserve_rng_state=False)
    else: assert checkpoint_type.startswith("matepoint"), f"checkpoint_type must be 'none', 'torch', or 'matepoint', got {checkpoint_type}"
    # The cuda stream needs to be created only once per process. This handles this for you.
    #
    # NOTE: we used to have these be global in the model files, but it caused some annoying things. I realized that I think it works being global in the 
    # matepoint file, but you have to use import matepoint and then matepoint.Gmatepoint_stream and stuff. Otherwise, it imports a reference. Ask an LLM if you are confused.
    if matepoint.Gmatepoint_stream is None:
        matepoint.Gmatepoint_stream = torch.cuda.Stream()

    o = matepoint.checkpoint(module, *args, **kwargs, use_reentrant=False, preserve_rng_state=False)
    if "sync" in checkpoint_type: torch.cuda.synchronize()
    return o

    

def rand_subset(x, dim=0, N=100_000):
    N = min(N, x.shape[dim])
    perm = torch.randperm(x.shape[dim])[:N]
    
    # Create slice tuple that works for all dimensions
    slc = [slice(None)] * x.dim()
    slc[dim] = perm
    return x[tuple(slc)]
