import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam

# vgg choice
base = {'vgg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

# vgg16
def vgg(cfg, i=3, batch_norm=True):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


def hsp(in_channel, out_channel):
    layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                           nn.ReLU())
    return layers

def cls_modulation_branch(in_channel, hiden_channel):
    layers = nn.Sequential(nn.Linear(in_channel, hiden_channel),
                           nn.ReLU())
    return layers

def cls_branch(hiden_channel, class_num):
    layers = nn.Sequential(nn.Linear(hiden_channel, class_num),
                           nn.Sigmoid())
    return layers

def concat_r():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.ConvTranspose2d(512, 512, 4, 2, 1)]
    return layers

def concat_1():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    return layers

def mask_branch():
    layers = []
    layers += [nn.Conv2d(512, 2, 3, 1, 1)]
    layers += [nn.ConvTranspose2d(2, 2, 8, 4, 2)]
    layers += [nn.Softmax2d()]
    return layers

def incr_channel():
    layers = []
    layers += [nn.Conv2d(128, 512, 3, 1, 1)]
    layers += [nn.Conv2d(256, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    return layers

def incr_channel2():
    layers = []
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    return layers

def norm(x, dim):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    normed = x / torch.sqrt(squared_norm)
    return normed

def fuse_hsp(x, p):
    group_size = 5
    t = torch.zeros(group_size, x.size(1))
    for i in range(x.size(0)):
        tmp = x[i, :]
        if i == 0:
            nx = tmp.expand_as(t)
        else:
            nx = torch.cat(([nx, tmp.expand_as(t)]), dim=0)
    nx = nx.view(x.size(0)*group_size, x.size(1), 1, 1)
    y = nx.expand_as(p)
    return y


class Model(nn.Module):
    def __init__(self, device, base, incr_channel, incr_channel2, hsp1, hsp2, cls_m, cls, concat_r, concat_1, mask_branch):
        super(Model, self).__init__()
        self.base = nn.ModuleList(base)
        self.sp1 = hsp1
        self.sp2 = hsp2
        self.cls_m = cls_m
        self.cls = cls
        self.incr_channel1 = nn.ModuleList(incr_channel)
        self.incr_channel2 = nn.ModuleList(incr_channel2)
        self.concat4 = nn.ModuleList(concat_r)
        self.concat3 = nn.ModuleList(concat_r)
        self.concat2 = nn.ModuleList(concat_r)
        self.concat1 = nn.ModuleList(concat_1)
        self.mask = nn.ModuleList(mask_branch)
        self.extract = [13, 23, 33, 43]
        self.device = device
        self.group_size = 5

    def forward(self, x):
        # backbone, p is the pool2, 3, 4, 5
        p = list()
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                p.append(x)

        # increase the channel
        newp = list()
        for k in range(len(p)):
            np = self.incr_channel1[k](p[k])
            np = self.incr_channel2[k](np)
            newp.append(self.incr_channel2[4](np))

        # spatial modulator
        spa_mask = spatial_optimize(newp[3], self.group_size).to(self.device)

        # hsp
        x = newp[3]
        x = self.sp1(x)
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        x = torch.bmm(x, x.transpose(1, 2))
        x = x.view(-1, x.size(1) * x.size(2))
        x = x.view(x.size(0) // 5, x.size(1), -1, 1)
        x = self.sp2(x)
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        x = torch.bmm(x, x.transpose(1, 2))
        x = x.view(-1, x.size(1) * x.size(2))

        #cls pred
        cls_modulated_vector = self.cls_m(x)
        cls_pred = self.cls(cls_modulated_vector)

        #semantic and spatial modulator
        g1 = fuse_hsp(cls_modulated_vector, newp[0])
        g2 = fuse_hsp(cls_modulated_vector, newp[1])
        g3 = fuse_hsp(cls_modulated_vector, newp[2])
        g4 = fuse_hsp(cls_modulated_vector, newp[3])

        spa_1 = F.interpolate(spa_mask, size=[g1.size(2), g1.size(3)], mode='bilinear')
        spa_1 = spa_1.expand_as(g1)
        spa_2 = F.interpolate(spa_mask, size=[g2.size(2), g2.size(3)], mode='bilinear')
        spa_2 = spa_2.expand_as(g2)
        spa_3 = F.interpolate(spa_mask, size=[g3.size(2), g3.size(3)], mode='bilinear')
        spa_3 = spa_3.expand_as(g3)
        spa_4 = F.interpolate(spa_mask, size=[g4.size(2), g4.size(3)], mode='bilinear')
        spa_4 = spa_4.expand_as(g4)

        y4 = newp[3] * g4 + spa_4
        for k in range(len(self.concat4)):
            y4 = self.concat4[k](y4)
        y3 = newp[2] * g3 + spa_3

        for k in range(len(self.concat3)):
            y3 = self.concat3[k](y3)
            if k == 1:
                y3 = y3 + y4
        y2 = newp[1] * g2 + spa_2

        for k in range(len(self.concat2)):
            y2 = self.concat2[k](y2)
            if k == 1:
                y2 = y2 + y3
        y1 = newp[0] * g1 + spa_1

        for k in range(len(self.concat1)):
            y1 = self.concat1[k](y1)
            if k == 1:
                y1 = y1 + y2
        y = y1

        # decoder
        for k in range(len(self.mask)):
            y = self.mask[k](y)
        mask_pred = y[:, 0, :, :]

        return cls_pred, mask_pred



# build the whole network
def build_model(device):
    return Model(device,
                 vgg(base['vgg']),
                 incr_channel(),
                 incr_channel2(),
                 hsp(512, 64),
                 hsp(64**2, 32),
                 cls_modulation_branch(32**2, 512),
                 cls_branch(512, 78),
                 concat_r(),
                 concat_1(),
                 mask_branch())

# weight init
def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def spatial_optimize(fmap, group_size):
    fmap_split = torch.split(fmap, group_size, dim=0)
    for i in range(len(fmap_split)):
        cur_fmap = fmap_split[i]
        with torch.no_grad():
            spatial_x = cur_fmap.permute(0, 2, 3, 1).contiguous().view(-1, cur_fmap.size(1)).transpose(1, 0)
            spatial_x = norm(spatial_x, dim=0)
            spatial_x_t = spatial_x.transpose(1, 0)
            G = spatial_x_t @ spatial_x - 1
            G = G.detach().cpu()

        with torch.enable_grad():
            spatial_s = nn.Parameter(torch.sqrt(245 * torch.ones((245, 1))) / 245, requires_grad=True)
            spatial_s_t = spatial_s.transpose(1, 0)
            spatial_s_optimizer = Adam([spatial_s], 0.01)

            for iter in range(200):
                f_spa_loss = -1 * torch.sum(spatial_s_t @ G @ spatial_s)
                spatial_s_d = torch.sqrt(torch.sum(spatial_s ** 2))
                if spatial_s_d >= 1:
                    d_loss = -1 * torch.log(2 - spatial_s_d)
                else:
                    d_loss = -1 * torch.log(spatial_s_d)

                all_loss = 50 * d_loss + f_spa_loss

                spatial_s_optimizer.zero_grad()
                all_loss.backward()
                spatial_s_optimizer.step()

        result_map = spatial_s.data.view(5, 1, 7, 7)

        if i == 0:
            spa_mask = result_map
        else:
            spa_mask = torch.cat(([spa_mask, result_map]), dim=0)

    return spa_mask



