# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:17:55 2021

@author: LEE
"""
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SEANet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, min_dim=16, strides = [2, 4, 5, 8],
                 causal = True, causal_delay=False, weight_norm = False, skip = False,  **kwargs):
        super().__init__()
        
        self.name = 'SEANet_%s_%d_%s%s'%('%sCau'%('Half' if causal_delay else 'Full') if causal else 'Ncau',
                                      min_dim,
                                      '+'.join(str(stride) for stride in strides),
                                      '_skip' if skip else ''
                                      )
        
        
        self.strides = strides
        self.min_dim = min_dim
        self.causal = causal
        self.causal_delay = causal_delay

        self.weight_norm = weight_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = skip
        self.conv_in = Conv1d(in_channels = in_channels,
                                 out_channels = min_dim,
                                 kernel_size = 7,
                                 stride = 1,
                                 bias = True, 
                                 activation = None, causality = causal, pre_activation=False)
                                
        
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*(2**i), min_dim*(2**(i+1)), stride, causal)
                                    for i, stride in enumerate(strides)])
        
        self.conv_bottle = nn.Sequential(
                                        Conv1d(in_channels=min_dim*(2**len(strides)),
                                                  out_channels = min_dim*(2**len(strides))//4,
                                                  kernel_size = 7, stride = 1,
                                                  activation = 'ReLU', causality= causal, pre_activation = True),
                                        
                                        Conv1d(in_channels=min_dim*(2**len(strides))//4,
                                                  out_channels = min_dim*(2**len(strides)),
                                                  kernel_size = 7, stride = 1,
                                                  activation = 'ReLU', causality= causal, pre_activation = True),
                                        )

        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*(2**(len(strides)-i)), min_dim*(2**(len(strides)-1-i)), stride, causal, causal_delay)
                                    for i, stride in enumerate(reversed(strides))])
        
        # self.decoder = nn.ModuleList([
        #                             DecBlock(min_dim*16, min_dim*8, 8, causal),
        #                             DecBlock(min_dim*8, min_dim*4, 8, causal),
        #                             DecBlock(min_dim*4, min_dim*2, 2, causal),
        #                             DecBlock(min_dim*2, min_dim, 2, causal),
        #                             ])
        
        self.conv_out = Conv1d(in_channels = min_dim,
                                   out_channels = out_channels,
                                   kernel_size = 7,
                                   stride = 1,
                                   bias= True,
                                   activation = 'ReLU', causality = causal, pre_activation=True)
        
        if weight_norm:
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    m = nn.utils.weight_norm(m)
        
        
    def forward(self, x):

        while len(x.size()) < 3:
            x = x.unsqueeze(-2)
        
        y = [x]
        
        x = self.conv_in(x)
        y.append(x)
        
        for encoder in self.encoder:
            # print(x.shape)
            x = encoder(x)
            #print(x.shape)
            y.append(x)
        # print(x.shape)
        x = self.conv_bottle(x)

        for l in range(len(self.decoder)):
            
            x = x[..., :y[-l-1].shape[-1]] + y[-l-1][..., :x.shape[-1]]
            
            x = self.decoder[l](x)
            # print(x.shape)
        x = x[..., :y[1].shape[-1]] + y[1][..., :x.shape[-1]]
        x = self.conv_out(x)
        
        
        if self.skip:
            x = x[..., :y[0].shape[-1]] + y[0][..., :x.shape[-1]]
        
        return x
    
    def get_name(self):
        return self.name
    
class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, causality=True):
        super().__init__()
        # self.res_unit = nn.Sequential(ResUnit(out_channels//2, 1, weight_norm, causality),
        #                               ResUnit(out_channels//2, 3, weight_norm, causality),
        #                               ResUnit(out_channels//2, 9, weight_norm, causality))
        
        self.refine = RNet(
                            channels = in_channels,
                            causality = causality
                            )
        
        self.conv = Conv1d(
                                 in_channels = in_channels, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 activation = 'ReLU',
                                 causality = causality, pre_activation=True
                                 )
        
    def forward(self, x):

        x = self.refine(x)
        x = self.conv(x)

        return x
        
    
class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, causality=True, causal_delay = True):
        super().__init__()

        
        self.conv = ConvTranspose1d(
                                 in_channels = in_channels, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 activation = 'ReLU',
                                 causality = (not causal_delay), pre_activation=True
                                 )
        
        self.refine = RNet(
                        channels = out_channels,
                        causality = causality
                        )
        
        # self.res_unit = nn.Sequential(ResUnit(out_channels, 1, weight_norm, causality),
        #                               ResUnit(out_channels, 3, weight_norm, causality),
        #                               ResUnit(out_channels, 9, weight_norm, causality))        
        self.stride = stride
        self.causality = causality
        
    def forward(self, x):
        x = self.conv(x)
        x = self.refine(x)
        return x

    
class RUnit(nn.Module):
    def __init__(self, channels, dilation = 1, causality=True):
        super().__init__()
        
        self.conv_in = Conv1d(
                                 in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 3, stride= 1,
                                 dilation = dilation,
                                 activation = 'ReLU',
                                 causality = causality, pre_activation=True
                                 )
        
        self.conv_out = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 activation = 'ReLU',
                                 causality = causality, pre_activation=True
                                 )
        
        
        self.pad = 2*dilation
        
    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_out(y)
        return x + y
        
    
    
class RNet(nn.Module):
    def __init__(self, channels, causality=True, CSP = False):
        super().__init__()    
        self.units = nn.Sequential(RUnit(channels, 1,  causality),
                                      RUnit(channels, 3,  causality),
                                      RUnit(channels, 9,  causality))
        
    def forward(self, x):
        return self.units(x)
        
    
    
            
class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size = 1, stride = 1, dilation = 1 , groups = 1,
                 causality=False, bias=False, 
                 activation = 'ReLU', pre_activation = False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride =stride,
                              dilation = dilation,
                              bias = bias,
                              groups = groups)
        
        self.activation = getattr(nn, activation)() if activation != None else Identity()
        self.pad = dilation * (kernel_size - 1) if causality else (dilation * (kernel_size - 1) + 1)//2
        self.pre_activation = pre_activation
        self.causality = causality

        
    def forward(self, x):
        
        if self.pre_activation:
            x = self.activation(x)
        
        x = self.conv(x)
        
        x = x[..., :-self.pad] if self.causality else x[..., self.pad:]
        
        if not self.pre_activation:
            x = self.activation(x)
        
        return x
                    
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 dilation = 1, groups = 1, bias= False, 
                 causality=False, 
                 activation='ReLU', pre_activation = False):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels,
                              kernel_size= kernel_size, stride= stride, 
                              dilation = dilation, groups = groups, bias= bias)
        
        self.pad = Pad(((kernel_size-1)*dilation, stride-1)) if causality else Pad((((kernel_size-1)*dilation)//2, ((kernel_size-1)*dilation)//2))
        
        self.activation = getattr(nn, activation)() if activation != None else Identity()
        #print(self.activation)
   
        self.pre_activation = pre_activation
        
    def forward(self, x):
        
        if self.pre_activation:
            x = self.activation(x)
        
        x = self.pad(x)
        x = self.conv(x)
        
        if not self.pre_activation:
            x = self.activation(x)
        return x

class Identity(nn.Module):
    def __init__(self, opt_print=False):
        super().__init__()
        self.opt_print = opt_print
    def forward(self, x):
        if self.opt_print: print(x.shape)
        return x
    
class Pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        return F.pad(x, pad=self.pad)    



if __name__ == "__main__":
    

    temp = torch.randn([2,1,16000])

    model = SEANet(in_channels=1, out_channels=1, min_dim = 16, strides = [2,4,5,8])
    
    output = model(temp)
    print(output.shape)
    model_paramteres = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_paramteres)
    

