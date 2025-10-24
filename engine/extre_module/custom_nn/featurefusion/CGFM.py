import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from engine.extre_module.ultralytics_nn.conv import Conv


class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):   
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(  
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
   
    def init_weights(self):   
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):   
                init.constant_(m.weight, 1)   
                init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear): 
                init.normal_(m.weight, std=0.001)    
                if m.bias is not None:    
                    init.constant_(m.bias, 0)    

    def forward(self, x):     
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)   
        y = self.fc(y).view(b, c, 1, 1) 
        return x * y.expand_as(x)
    
class MultiScaleEdgeEnhance(nn.Module):
    """
    多尺度边缘增强模块
    专门为水下模糊边界设计
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        
        reduced_ch = max(channels // reduction, 1)
        
        # 多尺度边缘检测
        self.edge_conv1 = nn.Conv2d(channels, reduced_ch, 3, 1, 1, groups=reduced_ch)  # 3x3
        self.edge_conv2 = nn.Conv2d(channels, reduced_ch, 5, 1, 2, groups=reduced_ch)  # 5x5
        self.edge_conv3 = nn.Conv2d(channels, reduced_ch, 7, 1, 3, groups=reduced_ch)  # 7x7
        
        # 边缘信息融合
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(reduced_ch * 3, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge1 = self.edge_conv1(x)
        edge2 = self.edge_conv2(x)  
        edge3 = self.edge_conv3(x)
        
        edge_concat = torch.cat([edge1, edge2, edge3], dim=1)
        edge_weight = self.edge_fusion(edge_concat)
        
        return x * (1 + edge_weight)  # 边缘增强

class AdaptiveUnderwaterFusion(nn.Module):

    def __init__(self, in_channels_list, out_channels, reduction=16):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_inputs = len(in_channels_list)
        
        if len(in_channels_list) == 2:
            inc0, inc1 = in_channels_list
            self.adjust_conv = nn.Identity()
            if inc0 != inc1:
                self.adjust_conv = Conv(inc0, inc1, k=1)
            target_channels = inc1
        else:
            # 支持多输入的情况
            target_channels = max(in_channels_list)
            self.adjust_convs = nn.ModuleList([
                Conv(inc, target_channels, k=1) if inc != target_channels else nn.Identity()
                for inc in in_channels_list
            ])
        
        self.target_channels = target_channels
        
        # 水下环境适应的SE注意力
        self.underwater_se = SEAttention(target_channels * self.num_inputs, reduction)
        
        # 多尺度边缘增强
        self.edge_enhance = MultiScaleEdgeEnhance(target_channels)
        
        # 输出维度调整
        concat_channels = target_channels * self.num_inputs
        if concat_channels != out_channels:
            self.conv1x1 = Conv(concat_channels, out_channels, k=1)
        else:
            self.conv1x1 = nn.Identity()

    def forward(self, feature_list):
        if len(feature_list) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} features, got {len(feature_list)}")
        
        # 1. 特征对齐和尺寸统一
        target_size = feature_list[0].shape[2:]
        
        if self.num_inputs == 2:
            # 双输入情况 - 完全借鉴ContextGuideFusionModule
            x0, x1 = feature_list
            
            # 尺寸对齐
            if x1.shape[2:] != target_size:
                x1 = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=False)
            
            # 通道对齐
            x0 = self.adjust_conv(x0)
            
            # 边缘增强
            x0 = self.edge_enhance(x0)
            x1 = self.edge_enhance(x1)
            
            aligned_features = [x0, x1]
        else:
            # 多输入情况
            aligned_features = []
            for feat, adjust_conv in zip(feature_list, self.adjust_convs):
                if feat.shape[2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                feat = adjust_conv(feat)
                feat = self.edge_enhance(feat)
                aligned_features.append(feat)
        
        # 2. 特征拼接 - 借鉴ContextGuideFusionModule
        x_concat = torch.cat(aligned_features, dim=1)
        
        # 3. 水下环境适应的SE注意力
        x_concat = self.underwater_se(x_concat)
        
        # 4. 双向增强融合 - 核心借鉴ContextGuideFusionModule的思想
        if self.num_inputs == 2:
            # 完全按照ContextGuideFusionModule的方式
            x0_weight, x1_weight = torch.split(x_concat, [self.target_channels, self.target_channels], dim=1)
            
            # 双向交互增强
            x0_enhanced = aligned_features[0] * x0_weight
            x1_enhanced = aligned_features[1] * x1_weight
            
            # 交叉融合 - ContextGuideFusionModule的精髓
            fused = torch.cat([
                aligned_features[0] + x1_weight,  # x0 + x1的权重
                aligned_features[1] + x0_weight   # x1 + x0的权重
            ], dim=1)
        else:
            # 多输入的扩展版本
            channel_splits = [self.target_channels] * self.num_inputs
            feature_weights = torch.split(x_concat, channel_splits, dim=1)
            
            enhanced_features = []
            for i, (feat, weight) in enumerate(zip(aligned_features, feature_weights)):
                # 计算其他特征权重的平均
                other_weights = [w for j, w in enumerate(feature_weights) if j != i]
                if other_weights:
                    cross_weight = sum(other_weights) / len(other_weights)
                    enhanced_feat = feat + cross_weight  # 交叉增强
                else:
                    enhanced_feat = feat
                enhanced_features.append(enhanced_feat)
            
            fused = torch.cat(enhanced_features, dim=1)
        
        # 5. 输出调整
        output = self.conv1x1(fused)
        return output



if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)
    
    print(BLUE + "=== Adaptive Underwater Fusion模块测试 ===" + RESET)
    print(GREEN + " AdaptiveUnderwaterFusion:" + RESET)
    auf = AdaptiveUnderwaterFusion([channel_1, channel_2], ouc_channel, reduction=16).to(device)
    
    outputs = auf([inputs_1, inputs_2])
    print(f'输入1: {inputs_1.size()}, 输入2: {inputs_2.size()}, 输出: {outputs.size()}')
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=auf,
                                   args=[[inputs_1, inputs_2]],
                                   output_as_string=True,
                                   output_precision=4,
                                   print_detailed=True)
    print(RESET)
    

    