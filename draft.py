import torch
from torch import nn

# import torch.nn as nn

x = torch.Tensor(3, 6, 3, 3)
print(x)

out = nn.Conv2d(in_channels=6,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=2,  # 边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
                bias=False)

tt = out(x)
print(tt)

# class model(nn.Module):
#     def __int__(self):
#         super(model, self).__int__()
#         self.conv1 = nn.Conv2d(in_channels=3,
#                                out_channels=3,
#                                kernel_size=3,
#                                stride=1,
#                                padding=2,  # 边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
#                                bias=False)
#
#     def forward(self, x):
#         output = self.conv1(x)
#
#         return output
#
#
# net = model()
