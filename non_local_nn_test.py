# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:52:38 2018

@author: P768978
"""

x = torch.rand(2,8,5,5)

phi_conv = nn.Conv2d(8,4,1)
theta_conv = nn.Conv2d(8,4,1)

theta_x = theta_conv(x)
phi_x = phi_conv(x)


a = torch.arange(2*2*3).reshape(2,2,3)
b = torch.arange(2*2*3).reshape(2,3,2)*2

#a
#Out[89]:
#tensor([[[ 0,  1,  2],
#         [ 3,  4,  5]],
#
#        [[ 6,  7,  8],
#         [ 9, 10, 11]]])
#
#b
#Out[90]:
#tensor([[[ 0,  2],
#         [ 4,  6],
#         [ 8, 10]],
#
#        [[12, 14],
#         [16, 18],
#         [20, 22]]])

torch.stack([torch.matmul(a[i], b[i]) for i in range(len(a))]).shape
#Out[91]:
#tensor([[[ 20,  26],
#         [ 56,  80]],
#
#        [[344, 386],
#         [488, 548]]])

torch.matmul(a,b)
#Out[112]:
#tensor([[[ 20,  26],
#         [ 56,  80]],
#
#        [[344, 386],
#         [488, 548]]])