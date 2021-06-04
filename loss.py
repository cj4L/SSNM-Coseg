from torch import nn
import torch.nn.functional as F
import torch


class Weighed_Bce_Loss(nn.Module):
    def __init__(self):
        super(Weighed_Bce_Loss, self).__init__()

    def forward(self, x, label):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        label = label.view(-1, 1, label.shape[1], label.shape[2])
        label_t = (label == 1).float()
        label_f = (label == 0).float()
        p = torch.sum(label_t) / (torch.sum(label_t) + torch.sum(label_f))
        w = torch.zeros_like(label)
        w[label == 1] = p
        w[label == 0] = 1 - p
        loss = F.binary_cross_entropy(x, label, weight=w)
        return loss


class Cls_Loss(nn.Module):
    def __init__(self):
        super(Cls_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.binary_cross_entropy(x, label)
        return loss

class S_Loss(nn.Module):
    def __init__(self):
        super(S_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.smooth_l1_loss(x, label)
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_wbce = Weighed_Bce_Loss()
        self.loss_cls = Cls_Loss()
        self.loss_s = S_Loss()
        self.w_wbce = 1
        self.w_cls = 1
        self.w_smooth = 1

    def forward(self, x, label, x_cls, label_cls):
        m_loss = self.loss_wbce(x, label) * self.w_wbce
        c_loss = self.loss_cls(x_cls, label_cls) * self.w_cls
        s_loss = self.loss_s(x, label) * self.w_smooth
        loss = m_loss + c_loss + s_loss

        return loss, m_loss, c_loss, s_loss

