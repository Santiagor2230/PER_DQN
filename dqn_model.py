import torch
from torch import nn
import numpy as np

class DQN(nn.Module):

  #obs_shape will be a tuple of = (channels, height, width)
  def __init__(self, hidden_size, obs_shape, n_actions): 
    super().__init__()
    # process visual information
    # Convolutional layers to reduce the channel = 64
    self.conv = nn.Sequential(
        nn.Conv2d(obs_shape[0], 64, kernel_size=3),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
    )
    conv_out_size = self._get_conv_out(obs_shape)

    self.head = nn.Sequential(
        nn.Linear(conv_out_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )

    self.fc_adv = nn.Linear(hidden_size, n_actions)
    self.fc_value = nn.Linear(hidden_size, 1)

  def _get_conv_out(self, shape):
    conv_out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(conv_out.size()))

  def forward(self, x):
    x = self.conv(x.float()).view(x.size()[0],-1)
    x = self.head(x)
    adv = self.fc_adv(x)
    value = self.fc_value(x)
    return value + adv - torch.mean(adv, dim=1, keepdim=True)
