import torch
import torch.nn as nn
class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride, initial=False):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect")
    self.instanceNorm = nn.InstanceNorm2d(out_channels)
    self.lrelu = nn.LeakyReLU(0.2)
    self.initial = initial

  def forward(self, x):
    x = self.conv(x)
    x = self.instanceNorm(x) if self.initial==False else x
    x = self.lrelu(x)
    return x

class Discriminator(nn.Module):
  def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
    super().__init__()
    layers=[]
    layers.append(CNNBlock(in_channels, features[0], 2, initial=True))
    in_channels = features[0]
    for feature in features[1:]:
      layers.append(CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2))
      in_channels = feature
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return torch.sigmoid(self.model(x))