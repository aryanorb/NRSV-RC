# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:05:44 2023

@author: gist
"""
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from sv_network        import ECAPA_TDNN, SEModule
from se_network        import SEANet

from ptflops import get_model_complexity_info

def initialize_weights(m):
  if isinstance(m, nn.Conv1d):
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        
        return x * y.expand_as(x)

    
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )
        
    def forward(self, input: torch.tensor):
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter)


class LISBlock(nn.Module):
    def __init__(self, channels):
        super(LISBlock,self).__init__()
        
        self.eps = 1e-6
    
        self.layer =  nn.Sequential(
            nn.Conv1d(channels*2, channels, kernel_size = 3,  padding = 1, bias = False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, enhanced, noisy, past_residual = None):
        
        residual = noisy - enhanced

        weight = torch.cat([enhanced, noisy], dim=1)
        weight = self.layer(weight)
        
        if past_residual == None:
            enhanced = enhanced + weight*residual
        else:
            enhanced = enhanced + weight*residual + (1-weight)*past_residual
            
        attenuate = 1 / (1 + weight + self.eps)
        
        return enhanced * attenuate, residual
        
        

class LIABlock(nn.Module):
    def __init__(self, channels):
        
        # Lost Information Aggregation Block
        
        super(LIABlock, self).__init__()
        
        self.mel_scale = 80
        ratio = 8
        inter_channel = int(channels // ratio)
        
        self.decoder_e = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.mel_scale, kernel_size=1, bias = False),
            nn.BatchNorm1d(self.mel_scale),
            nn.PReLU(),
        )
        
        self.decoder_n = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.mel_scale, kernel_size=1,  bias = False),
            nn.BatchNorm1d(self.mel_scale),
            nn.PReLU(),
        )
        
        self.decoder_r = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=self.mel_scale, kernel_size=1,  bias = False),
            nn.BatchNorm1d(self.mel_scale),
            nn.PReLU(),
        )
        
        self.global_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.mel_scale*2, inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_channel),
            nn.ReLU(),
            nn.Conv1d(inter_channel, self.mel_scale , kernel_size=1, bias=False),
            nn.BatchNorm1d(self.mel_scale),
        )
        
        self.local_layer = nn.Sequential(
            nn.Conv1d(self.mel_scale*2, inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_channel),
            nn.ReLU(),
            nn.Conv1d(inter_channel, self.mel_scale , kernel_size=1, bias=False),
            nn.BatchNorm1d(self.mel_scale),
        )
        
    def forward(self, enhanced, noisy, residual_sum):
        
        
        enhanced = self.decoder_e(enhanced)
        noisy    = self.decoder_n(noisy)

        residual_sum = self.decoder_r(residual_sum)
        
        residual = noisy - enhanced
        
        feature = torch.cat([residual, residual_sum], dim=1)
        
        feature_g = self.global_layer(feature)
        feature_l = self.local_layer(feature)
        
        weight = torch.sigmoid(feature_g + feature_l)
        
        estimated_R = weight*residual + (1-weight)*residual_sum
        
        return estimated_R


class GLIEBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(GLIEBlock,self).__init__()
        
        # Global and Local Information Extraction Module with dilated convolution
        
        self.ratio = 8
        
        self.kernel_size = 3
        
        self.dilation    = dilation
        
        inter_channel = int(channels // self.ratio)
        
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1)
    
        self.global_layer = nn.Sequential(
            nn.Conv1d(2*channels, inter_channel, kernel_size = 1, dilation = 1,  padding = 0, bias=False),
            nn.BatchNorm1d(inter_channel),
            nn.ReLU(),
            nn.Conv1d(inter_channel, channels , kernel_size = 1, dilation = 1,  padding = 0, bias=False),
            nn.BatchNorm1d(channels),
        )
        
        self.local_layer = nn.Sequential(
            nn.Conv1d(channels, inter_channel, kernel_size = 1, dilation = 1,  padding = 0, bias=False),
            nn.BatchNorm1d(inter_channel),
            nn.ReLU(),
            nn.Conv1d(inter_channel, channels , kernel_size = 1, dilation = 1,  padding = 0, bias=False),
            nn.BatchNorm1d(channels),
        )

        self.conv1x1 = nn.Conv1d(in_channels=3*channels, out_channels=channels, kernel_size=1, stride=1)
        
        
        self.dwconv = nn.Conv1d(in_channels=channels, out_channels=256, kernel_size=self.kernel_size, dilation= self.dilation,
                              padding = (self.kernel_size - 1) * dilation // 2,  groups=channels, bias=False)
        self.bn     = nn.BatchNorm1d(256)
        self.prelu  = nn.PReLU()
        
        self.pwconv = nn.Conv1d(in_channels = 256, out_channels = channels, kernel_size = 1, stride=1)
        

        self.se     = SELayer(channels, reduction=8)
        
    def forward(self, x):
        
        # extract global and local infomratoin
        
        gap = self.GAP(x)
        gmp = self.GMP(x)
        
        # gamp = cat(gap + gmp)
        gamp = torch.cat([gap,gmp],dim=1)
        
        global_features = self.global_layer(gamp)

        local_features  = self.local_layer(x)
        
        global_features = global_features.repeat(1,1,local_features.shape[-1])     
        
        features = torch.cat([x, global_features, local_features],dim=1) # [batch, 3*channel, frames]
        
        features = self.conv1x1(features)
        
        features = 0.5 * features + x
        
        residual = features.clone()
        
        features = self.dwconv(features)
        features = self.bn(features)
        features = self.prelu(features)
        
        features = self.pwconv(features)
        
        features = self.se(features)
        
        output = features + residual

        return output
    
class SpeechRestorationModule(nn.Module):
    def __init__(self, n_blocks, channels):
        super(SpeechRestorationModule, self).__init__()
        
        self.n_blocks = n_blocks
        self.channels = channels
        
        # for enhanced features
        self.encoder_e = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=self.channels, kernel_size=3, stride=1, padding = 1, bias = False),
            nn.BatchNorm1d(self.channels),
            nn.PReLU(),
        )
        
        # for noisy features
        self.encoder_n = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels = self.channels, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm1d(self.channels),
            nn.PReLU(),
        )
        
        self.GLIE_block_e = nn.ModuleList([GLIEBlock(self.channels, 2**i) for i in range(n_blocks)])
        self.GLIE_block_n = nn.ModuleList([GLIEBlock(self.channels, 2**i) for i in range(n_blocks)])
        
        self.LIS = nn.ModuleList([LISBlock(self.channels) for i in range(n_blocks)])
        self.LIA   = LIABlock(self.channels)
        
        
    def forward(self, enhanced, noisy):
        
        enhanced = self.encoder_e(enhanced) # [batch, 128, n_frames]
        noisy    = self.encoder_n(noisy) # [batch, 128, n_frames]
        
        residual_list = []

        for i in range(self.n_blocks):
            
            enhanced        = self.GLIE_block_e[i](enhanced)
            noisy           = self.GLIE_block_n[i](noisy)
            
            if i == 0:
                enhanced, residual        = self.LIS[i](enhanced, noisy, None)
                residual_list.append(residual)
            else:
                enhanced, residual        = self.LIS[i](enhanced, noisy, residual_list[i-1])
                residual_list.append(residual)

        residual_sum = torch.sum(torch.stack(residual_list,dim=-1),-1)
  
        estimated_R = self.LIA(enhanced, noisy, residual_sum)
        
        return estimated_R 

class NRSVRC(nn.Module):
    def __init__(self, **kwargs):
        super(NRSVRC, self).__init__()
        
        # SV configuration
        self.C                  = kwargs['C']
        self.spk_embedding_dim  = kwargs['spk_embedding_dim']
        self.n_mels             = kwargs['n_mels'] # 80
        self.n_blocks           = kwargs['n_blocks']
        
        # SRM
        self.channles           = kwargs['channles']
        
        self.mel_spec = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(
                sample_rate     = 16000,
                n_fft           = 512, 
                win_length      = 400, 
                hop_length      = 160, 
                window_fn       = torch.hamming_window,
                n_mels          = self.n_mels, 
                center          = False, 
                pad             = 120
            ),
        )
            
        # 1.Denosing process
        self.se = SEANet()
        
        # 2. Speech restoration module
        self.srm = SpeechRestorationModule(n_blocks = self.n_blocks, channels = self.channles)

        # 3. Speaker embedding extraction
        self.sv = ECAPA_TDNN(
            C                   = self.C,
            spk_embedding_dim   = self.spk_embedding_dim,
            n_mels              = self.n_mels #  mels or channels
        )
        
    def forward(self, noisy, aug):


        enhanced = self.se(noisy)
        
        if enhanced.isnan().any():
            print('enhanced', enhanced[0], enhanced.shape)
        
        enhanced_signal = enhanced.clone()
        
        with torch.no_grad():
            
            enhanced_feature = self.mel_spec(enhanced) + 1e-6
            noisy_feature    = self.mel_spec(noisy) + 1e-6
            
            # log scale
            enhanced_feature = enhanced_feature.squeeze(dim=1).log()
            noisy_feature    = noisy_feature.squeeze(dim=1).log()
            
            # normalize
            enhanced_feature    = enhanced_feature - torch.mean(enhanced_feature, dim=-1, keepdim=True)
            noisy_feature       = noisy_feature - torch.mean(noisy_feature, dim=-1, keepdim=True)

        estimated_R     = self.srm(enhanced_feature, noisy_feature)
        input_features  = enhanced_feature + estimated_R + noisy_feature

        spk_embedding   = self.sv(input_features, aug)
        
        return spk_embedding, enhanced_signal, estimated_R, enhanced_feature
    
if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device("cuda");print(device)
    else:
        raise RuntimeError('CUDA device is unavailable ... please check the CUDA')
    
    config = {
        'channles'   : 128,
        'n_blocks': 4,
        'C': 512,
        'spk_embedding_dim' : 256,
        'n_mels': 80,
    }
    
    model = NRSVRC(**config).to(device)
    
    def prepare_input(input_shape):
    
        inputs = {
            'noisy': torch.ones((1, *input_shape)).to(device),
            'aug': True
        }
        
        return inputs

    # 2-seconds
    macs, params = get_model_complexity_info(model, (1, 32000), as_strings=True, backend='pytorch',
                                     print_per_layer_stat=True, verbose=True, input_constructor = prepare_input)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        