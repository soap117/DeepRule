import torch
import torch.nn as nn
#from DCNP.modules import DeformConv
class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class cls(nn.Module):
    def __init__(self, k, inp_dim, out_dim, cat_num, stride=1, with_bn=True):
        super(cls, self).__init__()

        pad = (k - 1) // 2
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride),
                              bias=not with_bn)
        self.bn1 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, padding=(pad, pad))
        self.conv2 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride),
                               bias=not with_bn)
        self.bn2 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.pool2 = nn.MaxPool2d(2, 2, padding=(pad, pad))
        self.final = fully_connected(out_dim, cat_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.max(x, 2)[0]
        x = torch.max(x, 2)[0]
        final = self.final(x)
        return final

class line_cls(nn.Module):
    def __init__(self, inp_dim, cat_num):
        super(line_cls, self).__init__()
        self.mid_ = torch.nn.Linear(inp_dim, 256)
        self.final_ = fully_connected(256, cat_num)
    def forward(self, x):
        fea_dim = x.size(2)
        x = x.view(-1, fea_dim)
        mid = torch.tanh(self.mid_(x))
        final = self.final_(mid)
        return final

class offset(nn.Module):
    def __init__(self, k, inp_dim, out_dim, cat_num, stride=1, with_bn=True):
        super(offset, self).__init__()

        pad = (k - 1) // 2
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride),
                              bias=not with_bn)
        self.bn1 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2, 2, padding=(pad, pad))
        self.conv2 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride),
                               bias=not with_bn)
        self.bn2 = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.pool2 = nn.AvgPool2d(2, 2, padding=(pad, pad))
        self.final = fully_connected(out_dim, cat_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.max(x, 2)[0]
        x = torch.max(x, 2)[0]
        final = self.final(x)
        return final

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)

        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)
'''
class dcn(nn.Module):
    def __init__(self, num_deformable_groups=4, inC=256, outC=256, kH=3, kW=3):
        super(dcn, self).__init__()

        self.get_offset = nn.Conv2d(
            inC,
            num_deformable_groups * 2 * kH * kW,
            kernel_size=(kH, kW),
            stride=(1, 1),
            padding=(1, 1),
            bias=False).cuda()

        self.conv_offset2d = DeformConv(
            inC,
            outC, (kH, kW),
            stride=1,
            padding=1,
            deformable_groups=num_deformable_groups).cuda()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.get_offset(x)
        x = self.relu(self.conv_offset2d(x, offset))
        return x
'''

cls_ = cls(2, 256, 256, 5, stride=2)
temp = torch.zeros(2, 256, 128, 128)
ans = cls_(temp)