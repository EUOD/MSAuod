import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   

import torch, math
import torch.nn as nn
     
from engine.extre_module.ultralytics_nn.conv import Conv   

class MSDC(nn.Module):    
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True): 
        super(MSDC, self).__init__()
 
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
  
        self.dwconvs = nn.ModuleList([  
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)   
            )  
            for kernel_size in self.kernel_sizes     
        ])   
   
    def forward(self, x):   
        # Apply the convolution layers in a loop
        outputs = []     
        for dwconv in self.dwconvs: 
            dw_out = dwconv(x)
            outputs.append(dw_out) 
            if self.dw_parallel == False:
                x = x+dw_out  
        # You can return outputs based on what you intend to do with them    
        return outputs
    
class MSCB(nn.Module):    
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super(MSCB, self).__init__()  
        
        self.in_channels = in_channels
        self.out_channels = out_channels  
        self.stride = stride    
        self.kernel_sizes = kernel_sizes  
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel   
        self.add = add    
        self.n_scales = len(self.kernel_sizes)    
        # check stride value   
        assert self.stride in [1, 2]   
        # Skip connection if stride is 1    
        self.use_skip_connection = True if self.stride == 1 else False    

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)     
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            Conv(self.in_channels, self.ex_channels, 1)     
        )  
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:   
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(    
            # pointwise convolution   
            Conv(self.combined_channels, self.out_channels, 1, act=False)
        ) 
        if self.use_skip_connection and (self.in_channels != self.out_channels):    
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)     

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:    
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:     
            if self.in_channels != self.out_channels:     
                x = self.conv1x1(x)
            return x + out
        else:
            return out 
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups     
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)    
        return x
    
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = MSCB(in_channel, out_channel, kernel_sizes=[1, 3, 5]).to(device)
    
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)    
    print(RESET)     
