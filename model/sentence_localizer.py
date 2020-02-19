"""
Anchor Based Sentence Localizer
"""
from itertools import chain
from global_config import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networks.attention import AttentionMean, AttentionType0, AttentionType1, AttentionType2
from .networks.seq_encoder import RNNSeqEncoder
from utils.helper_function import hidden_transpose, se2cw


class FeatureMixerType0(nn.Module):
    """
    Localize text conditioned on video.
    """
    def __init__(self, hidden_dim, video_attention, text_attention, dropout):
        """
        """
        super(FeatureMixerType0, self).__init__()
        self.video_attention = video_attention
        self.text_attention = text_attention
        self.text_batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.video_batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*3, bias=False),
            nn.BatchNorm1d(hidden_dim*3),
            nn.ReLU(),
            nn.Dropout(p=dropout),   #让部分神经元以一定的概率P停止工作
            nn.Linear(hidden_dim*3, hidden_dim*3, bias=False),
            nn.BatchNorm1d(hidden_dim*3),
            nn.ReLU(),
        )

    def forward(self, video_encoding, video_last_hidden, video_mask, sent_encoding, sent_last_hidden, sent_mask):
        """
        video_encoding: (6,234,512)
        video_last_hidden: (6,1024)
        video_mask: (6,234,1)
        captions_encoding: (6,22,512)
        captions_last_hidden: (6,1024)
        sent_mask: (6,22,1)
        :return (batch, hidden_dim)
        """
        text_context, _ = self.text_attention(sent_encoding, video_last_hidden, sent_mask)   # (6,512) , (6,22)
        video_context, _ = self.video_attention(video_encoding, sent_last_hidden, video_mask) # (6,512) , (6,234)

        text_context = self.text_batch_norm(text_context)
        video_context = self.video_batch_norm(video_context)

        # combiner
        return self.combiner(torch.cat([self.fc(torch.cat([text_context, video_context], dim=1)),
                                        text_context*video_context,
                                        text_context+video_context], dim=1))  # fc(v, t), v+t, v*t


class SentenceLocalizer(nn.Module):
    """
    attention_type_sl=type0  
    regressor_scale=0.3  
    feature_mixer_type=type0
    video_use_residual=True 
    sent_use_residual=False
    pe_video=100
    pe_sent=10 
    ANCHOR_LIST = [

    # more global component
    [1./4, 1./2], [2./4, 1./2], [3./4, 1./2],  # 2, w=1/2
    [1./6, 1./3], [3./6, 1./3], [5./6, 1./3],  #[1./3, 2./3], [2./3, 2./3], # 3, w=1/3
    #  localize component
    [1./16, 1./8], [3./16, 1./8], [5./16, 1./8], [7./16, 1./8], [9./16, 1./8], [11./16, 1./8], [13./16, 1./8], [15./16, 1./8], # 4, w=1./8
    [1./2, 1.0]
]
    """    
    def __init__(self, hidden_dim, rnn_layer, rnn_cell, rnn_dropout, bidirectional, attention_type, scale,
                 sent_vocab_size, sent_embedding_dim, video_feature_dim, fc_dropout, anchor_list, feature_mixer_type,
                 video_use_residual, sent_use_residual, pe_video=100, pe_sent=20):
        """
        :param anchor_list: python list contain anchors. in CW format ranging in [0, 1)
        """
        super(SentenceLocalizer, self).__init__()
        assert bidirectional is False, "We do not support bidirectional RNN for model structure."
        self.scale = scale  # the final delta refining will be (-scale, scale)
        # video encoder
        self.video_encoder = RNNSeqEncoder(video_feature_dim, hidden_dim, rnn_cell, rnn_layer,
                                           bidirectional, rnn_dropout, video_use_residual)

        # text encoder
        self.sent_embedding = nn.Embedding(sent_vocab_size, sent_embedding_dim)  #(6000,512)
        self.text_encoder = RNNSeqEncoder(sent_embedding_dim, hidden_dim, rnn_cell, rnn_layer,
                                          bidirectional, rnn_dropout, sent_use_residual)

        # resolved hidden for attention layer
        resolved_hidden_dim = hidden_dim*rnn_layer*(2 if bidirectional else 1)

        # feature mixer TODO: add more feature mixer
        assert feature_mixer_type == 'type0'    # 参考论文公式(7),(8),(9)对编码后的特征进行融合
        attention_module = None
        if attention_type == 'mean':
            attention_module = AttentionMean
        elif attention_type == 'type0':
            attention_module = AttentionType0
        elif attention_type == 'type1':
            attention_module = AttentionType1
        elif attention_type == 'type2':
            attention_module = AttentionType2
        video_attention = attention_module(hidden_dim, resolved_hidden_dim, pe_video)  #论文公式(7)
        text_attention = attention_module(hidden_dim, resolved_hidden_dim, pe_sent)  #论文公式(8)
        self.feature_mixer = FeatureMixerType0(hidden_dim, video_attention, text_attention, fc_dropout) #论文公式(9)

        # anchor predictor & refiner
        self.anchor = Variable(FloatTensor(anchor_list))  # anchor_number, 2
        anchor_number = self.anchor.size(0)
        self.predictor = nn.Linear(hidden_dim*3, anchor_number)  # 预测每个预设anchor的得分
        self.refiner = nn.Linear(hidden_dim*3, 2*anchor_number)  # (c, w) 预测每个预设anchor的偏移值 

    def forward(self, video_feat, video_length, video_mask, sent, sent_length, sent_mask, sent_gather_idx):
        """
        :param video_feat:      (batch_video, length_video, feature_dim)
        :param video_length:    (batch_video, 2)
        :param video_mask:      (batch_video, length_video, 1)
        :param sent:            (batch_sent, length_sent)
        :param sent_length:     (batch_sent)
        :param sent_mask:       (batch_sent, length_sent)
        :param sent_gather_idx: (batch_sent)
        :return:
        """

        # feature encoding
        sent_embedding = self.sent_embedding(sent) # (6,22)-->(6,22,512)
        sent_feature, sent_hidden = self.text_encoder(sent_embedding) # (6,22,512) / (2,6,22,512)
        video_feature, video_hidden = self.video_encoder(video_feat)  # (6,234,512) / (2,6,234,512)

        # convert batch video to batch caption
        video_feature = video_feature.index_select(dim=0, index=sent_gather_idx) 
        video_hidden = video_hidden.index_select(dim=1, index=sent_gather_idx)
        video_seq_len, video_time_len = video_length.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)
        video_seq_len = video_seq_len.contiguous().long()  # (6,1)
        video_time_len = video_time_len.contiguous()
        video_mask = video_mask.index_select(dim=0, index=sent_gather_idx)  # (6,234,1)

        # fetch the last valid hidden state，取最后一个隐状态是因为最后一个隐状态包含了所有的信息
        sz0, sz1, sz2, sz3 = video_hidden.size()  # 2,6,234,512
        video_last_hidden = video_hidden.gather(dim=2,
            index=video_seq_len.view(1, -1, 1, 1).repeat(sz0, 1, 1, sz3)-1).squeeze(2) # (2,6,512)
        sz0, sz1, sz2, sz3 = sent_hidden.size()
        sent_last_hidden = sent_hidden.gather(dim=2,
            index=sent_length.view(1, -1, 1, 1).repeat(sz0, 1, 1, sz3)-1).squeeze(2)  # (2,6,512)

        # mix features
        video_last_hidden, sent_last_hidden = hidden_transpose(video_last_hidden), hidden_transpose(sent_last_hidden) # (2,6,512)-->(6,1024)
        mixed_feature = self.feature_mixer(video_feature, video_last_hidden, video_mask,
                                           sent_feature, sent_last_hidden, sent_mask)  # (6,1536)

        # localizing & refining
        score = self.predictor(mixed_feature)  # (6,1536)-->(6,15)
        refining = (F.sigmoid(self.refiner(mixed_feature)) * self.scale).view(mixed_feature.size(0), -1, 2)  #(6,15,2) scale=0.3
        # print(refining)
        delta_c, delta_w = refining.chunk(2, dim=2)  #(6,15,1), (6,15,1)
        delta_c = delta_c - self.scale / 2 # delfa_c is normalized, while delta_w not(normalized delta_w leads into small segment)
        _, prediction = score.max(1)  # whether this step is differentiable?  (6)  [14,13,3,5,8,10]
        
        
        # 选取得分最高的anchor
        anchor_c, anchor_w = self.anchor.index_select(index=prediction, dim=0).chunk(2, dim=1)  # batch, 1
        delta_c = delta_c.gather(dim=1, index=prediction.view(-1, 1, 1)).squeeze(1)  # batch, 1
        delta_w = delta_w.gather(dim=1, index=prediction.view(-1, 1, 1)).squeeze(1)  # batch, 1
        final_c = anchor_c + delta_c # (6,1)
        final_w = anchor_w + delta_w # (6,1)

        final_prediction_time = self.segment_resolver(final_c, final_w, video_time_len) # (6,2)

        return score, refining, final_prediction_time  # (6,15), (6,15,2), (6,2)

    def forward_eval(self, video_feat, video_length, video_mask, sent, sent_length, sent_mask, sent_gather_idx):

        _, _, ts_time = self.forward(video_feat, video_length, video_mask, sent, sent_length, sent_mask, sent_gather_idx)
        _, video_time_len = video_length.index_select(dim=0, index=sent_gather_idx).chunk(2, dim=1)
        ts_se = ts_time / video_time_len  # se in 0, 1
        return se2cw(ts_se)
        # c, w = se2cw(ts_se).chunk(2, dim=1)
        # return torch.cat([c,w], dim=1)

    def forward_diff(self, video_feat, video_length, video_mask, sent, sent_length, sent_mask, sent_gather_idx):
        score , refining, _ = self.forward(video_feat, video_length, video_mask, sent, sent_length, sent_mask, sent_gather_idx)
        _, prediction = score.max(1)
        delta_c, delta_w = refining.chunk(2, dim=2)  # batch, n_anchor, 1
        anchor_c, anchor_w = self.anchor.index_select(index=prediction, dim=0).chunk(2, dim=1)  # batch, 1
        delta_c = delta_c.gather(dim=1, index=prediction.view(-1, 1, 1)).squeeze(1)  # batch, 1
        delta_w = delta_w.gather(dim=1, index=prediction.view(-1, 1, 1)).squeeze(1)  # batch, 1
        final_c = anchor_c + delta_c
        final_w = anchor_w + delta_w
        return torch.cat([final_c, final_w], dim=1)  # batch, 2 & differential

    def get_parameter_group(self, params):
        return [
            {'name': 'sl_default',
             'params': self.parameters(),
             'lr': params['lr']
            },
        ]


    def get_parameter_group_c(self, params):
        # return [
        #     {'name': 'sl_regressor',
        #      'params': self.parameters(),
        #      'lr': params['lr'] / 10
        #     }
        # ]

        return [
            {'name': 'sl_regressor',
             'params': self.refiner.parameters(),
             'lr': params['lr']
            },
            {'name': 'sl_others',
             'params': chain(self.video_encoder.parameters(), self.text_encoder.parameters(),
                             self.sent_embedding.parameters(), self.feature_mixer.parameters(),
                             self.predictor.parameters()),
             'lr': params['lr'] / 10
            }
        ]

    def segment_resolver(self, c, w, length):
        """
        :param c:  (batch, 1) range in [0, 1]
        :param w:  (batch, 1) range in [0, 1]
        :param length: (batch, 1)
        :return: (batch, 2)
        """
        s = c - w / 2
        e = c + w / 2
        s, _ = torch.cat([s, torch.zeros_like(s)], dim=1).max(1)
        e, _ = torch.cat([e, torch.ones_like(e) - DELTA], dim=1).min(1)
        return torch.stack([s,e], dim=1) * length

    def build_loss(self, confidence_logits, regressing_result, segment_time, video_len_time, lambda_):
        """
        :param confidence_logits: (batch, n_anchor)
        :param regressing_result: (batch, n_anchor, 2)
        :param segment_time:      (batch, 2)
        :param video_len_time:    (batch)
        :param lambda_:            scalar
        :return:
        """
        label, delta_cw = self._construct_label(segment_time, video_len_time)
        confidence_loss = F.cross_entropy(confidence_logits, label)
        regress_logits = regressing_result.gather(dim=1,
            index=label.view(-1, 1, 1).expand(label.size(0), 1, 2)).squeeze(1)
        regress_loss = F.l1_loss(regress_logits, delta_cw)

        return confidence_loss + lambda_ * regress_loss

    def _construct_label(self, segment_time, video_len_time):
        """
        :param segment_time:    (batch, 2) in (s, e) format
        :param video_len_time:  (batch)
        :return:
            anchor_score: 1 if the anchor is the closest, otherwise 0
            delta_cw: range in [0, 1]
        """
        anchor_c, anchor_w = self.anchor.chunk(2, dim=1)  # n_anchor, 1
        anchor_se = torch.cat([anchor_c - anchor_w / 2, anchor_c + anchor_w / 2], dim=1)  # n_anchor, 2
        segment_se = segment_time / video_len_time.unsqueeze(1)  # batch, 2
        iou = Variable(self._compute_iou(segment_se.data, anchor_se.data))  # batch, n_anchor
        _, label = iou.max(1)  # batch
        delta_s, delta_e = (anchor_se.index_select(dim=0, index=label) - segment_se).chunk(2, dim=1)  # batch, 1
        delta_cw = torch.cat([(delta_s + delta_e) / 2, delta_e-delta_s], dim=1)  # batch, 2

        return label.detach(), delta_cw.detach()

    def _compute_iou(self, seg1, seg2):
        """
        :param seg1: (batch1, 2) in (s, e) format
        :param seg2: (batch2, 2) in (s, e) format
        :return: (batch1, batch2)
        """
        assert not isinstance(seg1, Variable)
        assert not isinstance(seg2, Variable)
        batch1, batch2 = seg1.size(0), seg2.size(0)
        seg1_s, seg1_e = seg1.unsqueeze(1).repeat(1, batch2, 1).chunk(2, dim=2) # batch1, batch2, 1
        seg2_s, seg2_e = seg2.unsqueeze(0).repeat(batch1, 1, 1).chunk(2, dim=2)  # batch1, batch2, 1

        min_end, _ = torch.cat([seg1_e, seg2_e], dim=2).min(2)
        max_end, _ = torch.cat([seg1_e, seg2_e], dim=2).max(2)
        min_beg, _ = torch.cat([seg1_s, seg2_s], dim=2).min(2)
        max_beg, _ = torch.cat([seg1_s, seg2_s], dim=2).max(2)
        intersection = min_end - max_beg
        intersection, _ = torch.stack([intersection, torch.zeros_like(intersection)], dim=2).max(2)  # batch1, batch2
        union = max_end - min_beg  # batch1, batch2
        iou = intersection / (union + DELTA)
        return iou

    def compute_mean_iou(self, seg1, seg2):
        """
        :param seg1: batch, 2 in (s, e) format
        :param seg2: batch, 2 in (s, e) format
        :return:
            miou: scalar
        """
        assert not isinstance(seg1, Variable)
        assert not isinstance(seg2, Variable)
        seg1_s, seg1_e = seg1.chunk(2, dim=1)  # batch, 1
        seg2_s, seg2_e = seg2.chunk(2, dim=1)  # batch, 1
        min_end, _ = torch.cat([seg1_e, seg2_e], dim=1).min(1)  # batch
        max_end, _ = torch.cat([seg1_e, seg2_e], dim=1).max(1)
        min_beg, _ = torch.cat([seg1_s, seg2_s], dim=1).min(1)
        max_beg, _ = torch.cat([seg1_s, seg2_s], dim=1).max(1)
        intersection = min_end - max_beg
        intersection, _ = torch.stack([intersection, torch.zeros_like(intersection)], dim=1).max(1)  # batch
        union = max_end - min_beg  # batch
        iou = intersection / (union + DELTA) # batch
        return iou.mean()
