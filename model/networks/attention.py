from math import sqrt

from global_config import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    """
    Attention Base Class
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=-1):
        """
        :param feature_dim: feature dimension of the input
        :param hidden_dim:  HERE, USE (hidden_dim * n_layers * n_directions) instead of your true hidden_dim
        """
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

    def forward(self, feature, hidden, mask):
        """
        :param feature: Float (batch, length, feature_dim)
        :param hidden:  Float (batch, hidden_dim), pls arrange the size of hidden to the specific format
        :param mask:    Float (batch, length, 1)
        :return:
            res:   Float (batch, feature_dim)
            alpha: Float (batch, length)
        """
        raise NotImplementedError()

#对应论文公式(12)
class AttentionMean(Attention):
    def forward(self, feature, hidden, mask, pe_size=-1):  # (B,T,512) (B,1024) (B,T,1)
        feature_masked_sum = torch.sum(feature * mask, dim=1)  # (B,512)
        feature_masked_weight = torch.sum(mask, dim=1) + DELTA  # (B,1)  DELTA=1e-4
        res = feature_masked_sum / feature_masked_weight
        return res, mask.squeeze(2)  # RES:(B,512) mask.squeeze:(B,T)

#论文公式(7),(8)对视觉特征和语句特征进行attention
class AttentionType0(Attention):
    """
    a_i = (f_i * W * hidden) / sqrt(d_k))
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=-1):
        super(AttentionType0, self).__init__(feature_dim, hidden_dim, pe_size)
        self.linear = nn.Linear(feature_dim, hidden_dim, bias=False)  # the matrix W: feature_dim * hidden_dim
    # feature:(6,22,512)  hidden: (6,1024) mask: (6,22,1)
    def forward(self, feature, hidden, mask):
        alpha= self.linear(feature)  # (6,22,512)-->(6,22,1024)
        #所有的通道相加求平均得到每一维的权重
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # (6,22,1024)
        alpha = alpha.sum(2, keepdim=True) / sqrt(self.hidden_dim)  # (6,22,1)
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')  # (6,22,1)
        alpha = alpha + mask_helper  # (6,22,1)
        alpha = F.softmax(alpha, dim=1)  # (6,22,1)
        res = (alpha * feature).sum(1)  # (6,512)
        return res, alpha.squeeze(2)  # (6,512) , (6,22)


class AttentionType1(Attention):
    """
    a_i = (f_i * W * hidden)
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=-1):
        super(AttentionType1, self).__init__(feature_dim, hidden_dim, pe_size)
        self.linear = nn.Linear(feature_dim, hidden_dim, bias=False)  # the matrix W: feature_dim * hidden_dim

    def forward(self, feature, hidden, mask):
        alpha = self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature_dim
        return res, alpha.squeeze(2)


class AttentionType2(Attention):
    """
    a_i = (f_i * W * hidden)
    advanced position aware attention module
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=100):
        super(AttentionType2, self).__init__(feature_dim, hidden_dim, pe_size)
        self.linear = nn.Linear(feature_dim, hidden_dim, bias=False)  # the matrix W: feature_dim * hidden_dim
        if pe_size != -1:
            self.pe_size = pe_size
            self.position_embedding = nn.Embedding(pe_size, feature_dim)
            nn.init.eye(self.position_embedding.weight)  # to be tested
            self.forward = self._forward_pe
        else:
            self.forward = self._forward

    def forward(self, feature, hidden, mask):
        raise NotImplementedError()

    def _forward_pe(self, feature, hidden, mask):
        seq_len = mask.sum(1)  # batch, 1
        pe_index = Variable((torch.arange(0, feature.size(1)).unsqueeze(0).repeat(feature.size(0), 1)).cuda())  # (batch, seq_len), float
        pe_index = pe_index / (seq_len + DELTA)  # (batch, seq_len), normalized to (0, 1+D)
        pe_index, _ = torch.stack([pe_index, torch.ones_like(pe_index) - DELTA], dim=2).min(2)  #  (0, 1-delta)
        pe_index = (pe_index * self.pe_size).long().detach()  # batch, seq_len
        feature = feature + self.position_embedding(pe_index)

        alpha= self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True) / sqrt(self.hidden_dim)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature
        return res, alpha.squeeze(2)


    def _forward(self, feature, hidden, mask):
        alpha = self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature_dim
        return res, alpha.squeeze(2)

# Attention With temporal segments
# 对应论文公式(11)
class ContextMaskC(nn.Module):
    def __init__(self, scale):  #scale=0.1
        super(ContextMaskC, self).__init__()
        self.scale = scale

    def forward(self, index, c, w):
        """
        :param index: Float (batch, length, 1)
        :param c:     Float (batch, 1, 1)
        :param w:     Float (batch, 1, 1)
        :return:
            masks: Float (batch, length, 1)
        """
        return F.sigmoid(self.scale * (index - c + w / 2)) - F.sigmoid(self.scale * (index - c - w / 2))  #scale是为了让sigmoid函数不陷入饱和区


class ContextMaskL(nn.Module):
    def __init__(self, scale):
        super(ContextMaskL, self).__init__()
        self.scale = scale

    def forward(self, index, c, w):
        """
        :param index: Float (batch, length, 1)
        :param c:     Float (batch, 1, 1)
        :param w:     Float (batch, 1, 1)
        :return:
            masks: Float (batch, length, 1)
        """
        return F.sigmoid(- self.scale * (index - c + w / 2))


class ContextMaskR(nn.Module):
    def __init__(self, scale):
        super(ContextMaskR, self).__init__()
        self.scale = scale

    def forward(self, index, c, w):
        """
        :param index: Float (batch, length, 1)
        :param c:     Float (batch, 1, 1)
        :param w:     Float (batch, 1, 1)
        :return:
            masks: Float (batch, length, 1)
        """
        return F.sigmoid(self.scale * (index- c - w / 2))

class AttentionMask(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_type, context_type, scale):
        """
        :param feature_dim: 512
        :param hidden_dim:  pls arrange the size of hidden to the specific format  512
        :param attention_type: mean
        :param context_type: clr
        :param scale: 0.1
        """
        super(AttentionMask, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.context_type = context_type.lower()
        if self.context_type == 'c':
            self.mask_c = ContextMaskC(scale)
            self.attention_c = self._build_attention(attention_type)
            self.forward = self._forwardc
        elif self.context_type == 'cl':
            self.mask_c = ContextMaskC(scale)
            self.attention_c = self._build_attention(attention_type)
            self.mask_r = ContextMaskR(scale)
            self.attention_r = self._build_attention(attention_type)
            self.forward = self._forwardcl
        elif self.context_type == 'clr':
            self.mask_c = ContextMaskC(scale)
            self.attention_c = self._build_attention(attention_type)  #attention_type=mean
            self.mask_r = ContextMaskR(scale)
            self.attention_r = self._build_attention(attention_type)
            self.mask_l = ContextMaskL(scale)
            self.attention_l = self._build_attention(attention_type)
            self.forward = self._forwardclr
        else:
            raise Exception('other attention types are not supported currently')

    def _build_attention(self, attention_type):
        if attention_type.lower() == 'mean':
            return AttentionMean(self.feature_dim, self.hidden_dim)  #(512,512)
        elif attention_type.lower() == 'type0':
            return AttentionType0(self.feature_dim, self.hidden_dim)
        elif attention_type.lower() == 'type1':
            return AttentionType1(self.feature_dim, self.hidden_dim)
        else:
            raise Exception('other attention types are not supported currently')

    def forward(self, feature, hidden, segment, mask):
        """
        :param feature: (batch, length, feature_dim)
        :param hidden:  (batch, hidden_dim)
        :param segment: (batch, 2)
        :param mask:    (batch, 2)
        :return:
            context
        """
        raise NotImplementedError()

    def _forwardc(self, feature, hidden, segment, mask):
        batch_size, seq_len = feature.size(0), feature.size(1)
        c, w = segment.unsqueeze(2).chunk(2, dim=1)  # batch, 1, 1
        mask_index = Variable(FloatTensor(range(seq_len)).expand(batch_size, seq_len).unsqueeze(2))  # batch_size, seq_len, 1
        c_context, _ = self.attention_c(feature, hidden, mask*self.mask_c(mask_index, c, w))
        return c_context

    def _forwardcl(self, feature, hidden, segment, mask):
        batch_size, seq_len = feature.size(0), feature.size(1)
        c, w = segment.unsqueeze(2).chunk(2, dim=1)  # batch, 1, 1
        mask_index = Variable(FloatTensor(range(seq_len)).expand(batch_size, seq_len).unsqueeze(2))  # batch_size, seq_len, 1
        l_context, _ = self.attention_l(feature, hidden, mask * self.mask_l(mask_index, c, w))
        c_context, _ = self.attention_c(feature, hidden, mask * self.mask_c(mask_index, c, w))
        return torch.cat([l_context, c_context], dim=1)

    """
    feature : (B,T,C)
    hidden: (B,num_layers*512)
    segment: (B,2)
    mask: (B,T,1)
    """
    def _forwardclr(self, feature, hidden, segment, mask):
        batch_size, seq_len = feature.size(0), feature.size(1)
        c, w = segment.unsqueeze(2).chunk(2, dim=1)  # batch, 1, 1
        mask_index = Variable(FloatTensor(range(seq_len)).expand(batch_size, seq_len).unsqueeze(2))  # (batch_size, seq_len, 1)
        l_context, _ = self.attention_l(feature, hidden, mask * self.mask_l(mask_index, c, w)) #(B,512)
        r_context, _ = self.attention_r(feature, hidden, mask * self.mask_r(mask_index, c, w)) #(B,512)
        c_context, _ = self.attention_c(feature, hidden, mask * self.mask_c(mask_index, c, w)) #(B,512)
        return torch.cat([l_context, c_context, r_context], dim=1)  # (B,3*512)
