import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

class StyleClassifier_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, p=0, style_dict={}, **kwargs):
    super().__init__()

    # Change default num of speaker from input to 25 if pretrained model (trainer.py - line 406 & 407) is activated (bug fixed)
    # out_feats = len(style_dict) # num of speakers (No IS)
    out_feats = 25 # (IS)

    self.classifier = nn.ModuleList()
    self.classifier.append(ConvNormRelu(in_channels, 64, downsample=True))
    self.classifier.append(ConvNormRelu(64, 128, downsample=True))
    self.classifier.append(ConvNormRelu(128, 128, downsample=True))
    self.classifier.append(ConvNormRelu(128, 256, downsample=True))
    self.classifier.append(ConvNormRelu(256, 256, downsample=True))
    self.classifier.append(ConvNormRelu(256, out_feats, downsample=True))
    self.model = nn.Sequential(*self.classifier)

  def forward(self, x, y, **kwargs):
    y_cap = self.model(x.transpose(-1, -2)).squeeze(-1)

    #internal_losses = [torch.nn.functional.cross_entropy(y_cap, y)]
    internal_losses = []
    
    return y_cap, internal_losses
