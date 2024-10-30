# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:02:31 2022

@author: gist
"""
import torch, math
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from auraloss.freq import MultiResolutionSTFTLoss, MelSTFTLoss


class MultiMelSTFTLoss(nn.Module):
    def __init__(self, device):
        super(MultiMelSTFTLoss, self).__init__()
        
        self.mel_filterbanks = [20,40,80]
        self.device = device
        
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        
        self.mel_stft_losses = []
        
        for mel_filter in self.mel_filterbanks:
            self.mel_stft_losses.append(
                MelSTFTLoss(
                    sample_rate     = self.sample_rate, 
                    fft_size        = self.n_fft,
                    win_length      = self.win_length,
                    hop_size        = self.hop_length,
                    n_mels          = mel_filter,
                    device          = self.device
                )
            )
        self.mel_stft_losses = nn.ModuleList(self.mel_stft_losses)

    def forward(self, clean, enhanced):
        
        loss = 0
        
        for mel_stft_loss in self.mel_stft_losses:
            
            loss += mel_stft_loss(clean, enhanced)
        
        loss = loss/len(self.mel_stft_losses)
        
        return loss


class MultiMelSpectrogramLoss(nn.Module):
    def __init__(self, device):
        super(MultiMelSpectrogramLoss, self).__init__()
        
        self.device = device
        self.mel_filterbanks = [20,40,80]
        
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.window_fn = torch.hamming_window
        
        self.mel_spectrograms = []
        
        for mel_filters in self.mel_filterbanks:
            self.mel_spectrograms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate     = self.sample_rate, 
                    n_fft           = self.n_fft,
                    win_length      = self.win_length,
                    hop_length      = self.hop_length,
                    window_fn       = self.window_fn,
                    n_mels          = mel_filters
                ).to(self.device)
            )
        self.mel_spectrograms = nn.ModuleList(self.mel_spectrograms)

        self.L1_loss = nn.L1Loss()
        
    def forward(self, clean, enhanced):
        
        loss = 0
        
        for mel_spectrogram in self.mel_spectrograms:
            clean_mel    = mel_spectrogram(clean)
            enhanced_mel = mel_spectrogram(enhanced)
            
            loss += self.L1_loss(clean_mel, enhanced_mel)
        
        loss = loss/len(self.mel_spectrograms)
        
        return loss

    
class ConsistencyLoss(nn.Module):
    def __init__(self, type):
        
        super(ConsistencyLoss, self).__init__()
        
        self.type = type
 
        self.KL_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, predicted, true):
        
        
        if self.type == 'L2':
            
            x = F.normalize(predicted, dim=1)
            y = F.normalize(true, dim=1)
            
            loss = 2 - 2 * (predicted * true).sum(dim=-1)
        
            return loss.mean()
        
        elif self.type == 'symKL':
            # KL (q||p)
            predicted_d     = nn.functional.softmax(predicted, dim=1) 
            true_log_d      = nn.functional.log_softmax(true, dim=1) 
            loss1           = self.KL_loss(true_log_d, predicted_d)
            
            # KL (p||q)
            predicted_log_d = nn.functional.log_softmax(predicted, dim=1)
            true_d          = nn.functional.softmax(true, dim=1) # prediction
            loss2           = self.KL_loss(predicted_log_d, true_d)
        
            loss            = loss1 + loss2
            
            return loss
        else:
            raise RuntimeError('check your consistency type: {}'.format(self.type))


# Exaggerated target speech resotration loss
class ETSRLoss(nn.Module):
    def __init__(self, device, reduction = True, eps = 1e-6):
        super(ETSRLoss,self).__init__()
        
        self.device = device
        self.l1_loss    = nn.L1Loss()
        
        self.reduction  = reduction
    
    def forward(self, estimated_R, clean_spec, enhanced_spec):
        
        # batch, mel_scale, n_frame
        epsilon = clean_spec - enhanced_spec
        
        epsilon_contiguous = epsilon.contiguous()  # .contiguous() 추가
        epsilon_max = epsilon_contiguous.view(epsilon.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        epsilon_min = epsilon_contiguous.view(epsilon.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        
        weight_matrix = torch.exp(epsilon / (epsilon_max - epsilon_min + 1e-8))
        
        weighted_residual = weight_matrix * epsilon
        
        loss = self.l1_loss(estimated_R, weighted_residual)
    
        return loss

# Asymmetric penality SR loss
class APSRLoss(nn.Module):
    def __init__(self, device, reduction=True, eps=1e-6):
        super(APSRLoss, self).__init__()
        
        self.device = device
        self.l1_loss = nn.L1Loss()
        self.reduction = reduction

    def forward(self, true_clean_spec, estimated_clean_spec):
        
        epsilon = estimated_clean_spec - true_clean_spec
        
        over_estimation = torch.where(epsilon >= 0, 0.5 * epsilon, torch.zeros_like(epsilon).to(self.device))
        under_estimation = torch.where(epsilon < 0, 2 * epsilon, torch.zeros_like(epsilon).to(self.device))
        
        loss = torch.abs(over_estimation + under_estimation)
        
        return loss.mean()


def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res

class AAMSoftmax(nn.Module):
    def __init__(self, n_class, m, s, embedding_size):
        
        super(AAMSoftmax, self).__init__()
        
        self.m = m
        self.s = s
        self.embedding_size     = embedding_size
        self.weight             = torch.nn.Parameter(torch.FloatTensor(n_class, self.embedding_size), requires_grad=True)
        self.ce                 = nn.CrossEntropyLoss()
        
        nn.init.xavier_normal_(self.weight, gain=1)
        
        self.cos_m  = math.cos(self.m)
        self.sin_m  = math.sin(self.m)
        self.th     = math.cos(math.pi - self.m)
        self.mm     = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine  = F.linear(F.normalize(x), F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output  = output * self.s
        
        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
    