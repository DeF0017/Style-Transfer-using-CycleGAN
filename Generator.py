import torch
import torch.nn as nn
class cnnBlock(nn.Module):
  def __init__(self, in_channels, out_channels, up_sample=False, use_act=True, **kwargs):
    super().__init__()
    self.cnn_block = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        if up_sample else
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding_mode="reflect", **kwargs),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True) if use_act else nn.Identity()
    )

  def forward(self, x):
    return self.cnn_block(x)

class residualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.resBlock = nn.Sequential(
        cnnBlock(channels, channels, kernel_size=3, padding=1),
        cnnBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
    )

  def forward(self, x):
    return x + self.resBlock(x)

class Generator(nn.Module):
  def __init__(self, img_channels=3, features=64, num_residual=9):
    super().__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
        nn.ReLU()
    )
    self.downBlock = nn.ModuleList([
        cnnBlock(features, features*2, kernel_size=3, stride=2, padding=1),
        cnnBlock(features*2, features*4, kernel_size=3, stride=2, padding=1)
    ])
    self.resBlock = nn.Sequential(*[residualBlock(features*4) for _ in range(num_residual)])
    self.upBlock = nn.ModuleList([
        cnnBlock(features*4, features*2, up_sample=True, kernel_size=3, stride=2, padding=1, output_padding=1),
        cnnBlock(features*2, features, up_sample=True, kernel_size=3, stride=2, padding=1, output_padding=1),
    ])
    self.final = nn.Conv2d(features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

  def forward(self, x):
    x = self.initial(x)
    for layer in self.downBlock:
      x = layer(x)
    x = self.resBlock(x)
    for layer in self.upBlock:
      x = layer(x)
    x = self.final(x)
    return torch.tanh(x)