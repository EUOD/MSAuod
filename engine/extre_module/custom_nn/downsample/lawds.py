import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch 
import torch.nn as nn
from einops import rearrange
from engine.extre_module.ultralytics_nn.conv import Conv, autopad    
   
  
class AWDS(nn.Module):
    # Light Adaptive-weight downsampling
    def __init__(self, in_ch, out_ch, group=16) -> None:   
        super().__init__()
        
        self.softmax = nn.Softmax(dim=-1)   
        self.attention = nn.Sequential(   
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), 
            Conv(in_ch, in_ch, k=1) 
        )     
        
        self.ds_conv = Conv(in_ch, in_ch * 4, k=3, s=2, g=(in_ch // group))   
        self.conv1x1 = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()   
  
    def forward(self, x):   
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4     
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)   
        att = self.softmax(att)     
        
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)   
        x = torch.sum(x * att, dim=-1)   
        return self.conv1x1(x)   
     
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = AWDS(in_channel, out_channel, group=16).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)