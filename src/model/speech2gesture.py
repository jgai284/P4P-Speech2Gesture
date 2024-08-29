# gan.py handles the higher-level training mechanics and integration, while speech2gesture.py specifies the detailed structure of the neural networks (G and D) involved

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *

# 'torch' is PyTorch library
import torch
# 'torch.nn' is a sub-module within PyTorch library that provides tools for building neural networks
import torch.nn as nn

# "nn.Module" is a base class in PyTorch for all neural network modules. When you define your own neural network model in PyTorch, you typically subclass "nn.Module" and define your layers and operations in the "__init__" method, and then specify how data should pass through these layers in the "forward" method.

# Translation model/generator G:
# A neural network called "Speech2Gesture_G" that generates temporal stack of 2D poses corresponding to the downsampled 1D signal
class Speech2Gesture_G(nn.Module):
  '''
  Baseline: http://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/

  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''

  # Constructor
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, **kwargs):
    super(Speech2Gesture_G, self).__init__()

    # Layers
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.style_encoder = StyleEncoder(in_channels, in_channels)
    self.residual_attention_block = ResidualAttentionBlock(in_channels, in_channels)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)
                                  for i in range(4)]))
    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

    self.embedding_layer = nn.Linear(6144, 128)

  # Define how the input audio features are processed through the network to produce the output gesture feature
  def forward(self, x, y, time_steps=None, **kwargs):
    if x.dim() == 3:
      x = x.unsqueeze(dim=1)
    x = self.audio_encoder(x, time_steps)
    x = self.unet(x)
    x = self.style_encoder(x)
    x = self.residual_attention_block(x)
    x = self.decoder(x)
    x = self.logits(x)

    internal_losses = []

    gesture_features = x.transpose(-1, -2)
    gesture_embeddings = self.get_embeddings(gesture_features)

    # swaps the last two dimensions of the tensor x
    # if x = (N, T, C), the result would be x = (N, C, T)
    # return x.transpose(-1, -2), internal_losses
    return gesture_features, gesture_embeddings, internal_losses
  
  def get_embeddings(self, gesture_features):
    # Flatten the features to prepare for fully connected layer
    batch_size, seq_length, num_features = gesture_features.size()
    flattened = gesture_features.reshape(batch_size, -1)
    embeddings = self.embedding_layer(flattened)
    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize the embeddings
    return embeddings
  
  def contrastive_loss(self, anchor, positive, negative, margin=1.0):
        pos_dist = nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_dist = nn.functional.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(F.relu(pos_dist - neg_dist + margin))
        return loss

  def compute_loss(self, speech_features, gesture_features):
      # Assuming speech_features and gesture_features are tensors of the same shape
      # Extract embeddings
      speech_embeddings = self.get_embeddings(speech_features)
      gesture_embeddings = self.get_embeddings(gesture_features)
      
      # Create positive and negative pairs
      # Example: for simplicity, using the same batch as positive pairs
      anchor = speech_embeddings
      positive = gesture_embeddings
      negative = gesture_embeddings  # This should ideally be from a different batch or source

      # Compute contrastive loss
      loss = self.contrastive_loss(anchor, positive, negative)
      return loss

# Input: (N, time, frequency) → After unsqueeze, becomes (N, 1, time, frequency).
# After AudioEncoder: (N, 256, time, 1).
# After UNet1D: (N, in_channels, time).
# After decoder: (N, in_channels, time).
# After logits: (N, out_feats, time).
# Output: (N, time, pose_feats) after transposing.



# Adversarial discriminator D: 
# A neural network called "Speech2Gesture_D" that provides scores to asses whether generated results are realistic
class Speech2Gesture_D(nn.Module):
  '''
  Baseline: http://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/

  input_shape:  (N, time, pose_feats)
  output_shape: (N, *, 1) ## discriminator scores
  '''

  # Constructor
  def __init__(self, in_channels=104, out_channels=64,  n_downsampling=2, p=0, groups=1, **kwargs):
    super(Speech2Gesture_D, self).__init__()

    self.conv1 = nn.Sequential(torch.nn.Conv1d(in_channels*groups, out_channels*groups, 4, 2, padding=1, groups=groups),
                               torch.nn.LeakyReLU(negative_slope=0.2))
    self.conv2 = nn.ModuleList([])
    for n in range(1, n_downsampling):
      ch_mul = min(2**n, 8)
      self.conv2.append(ConvNormRelu(out_channels, out_channels*ch_mul,
                                     type='1d', downsample=True, leaky=True, p=p, groups=groups))

    self.conv2 = nn.Sequential(*self.conv2)
    ch_mul_new = min(2**n_downsampling, 8)
    self.conv3 = ConvNormRelu(out_channels*ch_mul, out_channels*ch_mul_new,
                              type='1d', leaky=True, kernel_size=4, stride=1, p=p, groups=groups)

    out_shape = 1 if 'out_shape' not in kwargs else kwargs['out_shape']
    self.logits = nn.Conv1d(out_channels*ch_mul_new*groups, out_shape*groups, kernel_size=4, stride=1, groups=groups)

  # Defines how the input pose features are processed through the network to produce the discriminator scores
  def forward(self, x):
    x = x.transpose(-1, -2)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.logits(x)

    internal_losses = []
    return x.transpose(-1, -2).squeeze(dim=-1), internal_losses
