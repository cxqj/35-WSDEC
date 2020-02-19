from global_config import *
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helper_function import hidden_transpose

class RNNSeqDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, vocab_embedding_dim, attention_module, max_cap_length,
                 rnn_cell, n_layers, rnn_dropout, embedding=None):
        """
        :param input_dim: := vocab_embedding_dim + attention_model.output_dim
        """
        super(RNNSeqDecoder, self).__init__()
        if rnn_cell.lower() == 'rnn':
            self.rnn = nn.RNN
        elif rnn_cell.lower() == 'gru':
            self.rnn = nn.GRU
        else:
            raise Exception('other rnn_cells are not implemented currently')
        self.vocab_size = vocab_size
        self.max_cap_length = max_cap_length

        # rnn_cell
        self.rnn_cell = self.rnn(input_dim, hidden_dim, n_layers, dropout=rnn_dropout, batch_first=True)

        # embedding
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, vocab_embedding_dim) #(6000,512)

        # attention
        self.attention_module = attention_module

        # output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size) #512-->6000

    def forward(self, encoder_output, decoder_init_hidden, encoding_mask, temp_seg, ref_caption=None, beam_size=3):
        """
        :param encoder_output: (batch, length, feat_dim)
        :param decoder_init_hidden: (num_layers, batch, hidden_dim)
        :param encoding_mask: (batch, length, 1)
        :param temp_seg: (batch, 2)
        :param ref_caption: (batch, length_cap)
        :param beam_size: scalar
        :return:
            caption_prob:   for training, (batch, length_cap, vocab_size)
            caption_pred    for testing,  (batch, max_cap_len)
            caption_length: for testing,  (batch)
            caption_mask:   for testing,  (batch, max_cap_len, 1)
        """
        if self.training:
            return self.forward_train(encoder_output, decoder_init_hidden, encoding_mask, temp_seg, ref_caption)
        else:
            return self.forward_bmsearch(encoder_output, decoder_init_hidden, encoding_mask, temp_seg, beam_size)

    def forward_train(self, encoder_output, decoder_init_hidden, encoding_mask, temp_seg, ref_caption):
        batch_size = encoder_output.size(0)

        output_prob = []
        output_pred = []

        assert BOS_ID == 0
        hidden = decoder_init_hidden
        self.rnn_cell.flatten_parameters()
        assert (ref_caption[:, 0] == BOS_ID).all(), 'the first word is supposed to be <bos>'
        assert (ref_caption[:, -1] == EOS_ID).all(), 'the last work is supposed to be <eos>'

        # append <bos> to the output
        output_prob.append(Variable(torch.zeros(batch_size, 1, self.vocab_size) + BOS_ID).cuda()) # (B,1,vocab_size)
        output_pred.append(Variable(torch.zeros(batch_size, 1)).long().cuda() + BOS_ID) # (B,1)

        for i in range(ref_caption.size(1) - 1): # the last word is not fed into the net
            input_word_embedding = self.embedding(ref_caption[:, i]).unsqueeze(1)  # (B, 1, 512)
            context = self.attention_module(encoder_output, hidden_transpose(hidden), temp_seg, encoding_mask)   #(B,3*512)
            inputs = torch.cat([input_word_embedding, context.unsqueeze(1)], dim=2)  # (B,1,2048)
            rnn_output, hidden = self.rnn_cell(inputs, hidden)  # (B,1,512), (2,B,512)
            output = F.log_softmax(self.output_layer(rnn_output.squeeze(1)), dim=1) # (B,500)
            output_prob.append(output.unsqueeze(1)) #(B,1,500)
            _, next_input_word = output.max(1)  # B
            output_pred.append(next_input_word.unsqueeze(1)) # (B,1)
        return torch.cat(output_prob, dim=1), torch.cat(output_pred, dim=1), None, None


   """
    :param  encoder_output: (B,T,C)  note that B is captions number when training B=batch_size but test B=captions number
    :param  decoder_init_hidden: (2,B,C)
    :param  encoding_mask: (B,T,1)
    :param  temp_seg: (B,2)
    :return
       
    """
    def forward_bmsearch(self, encoder_output, decoder_init_hidden, encoding_mask, temp_seg, beam_size):

        batch_size = encoder_output.size(0)
        self.rnn_cell.flatten_parameters()
        out_pred_target_list = list()  # [(batch, k),..]  记录得分结果
        out_pred_parent_list = list()  # [(batch, k),..]  记录是target的结果属于哪一个beam_size
        candidate_score_dict = dict()  # [set()]  # 全是-inf??

        current_scores = FloatTensor(batch_size, beam_size).zero_() # (B,3)

        # out_pred_target_list.append(Variable(torch.zeros(batch_size, beam_size).long().cuda()) + BOS_ID)  # append (batch_size, k) <bos> to the pred list
        # out_pred_parent_list.append(Variable(torch.zeros(batch_size, beam_size).long().cuda()) - 1)  # append (batch_size, k) -1 to the pred list

        out_pred_target_list.append(Variable(torch.zeros(batch_size, beam_size).long()) + BOS_ID)  # (B,3)  [[0,0,0],...[0,0,0]]
        out_pred_parent_list.append(Variable(torch.zeros(batch_size, beam_size).long()) - 1)  # (B,3) [[-1,-1,-1],...[-1,-1,-1]]

        current_scores[:, 1:].fill_(-float('inf'))  #(B,3) [[0,-inf,-inf],...[0,-inf,-inf]]]
        current_scores = Variable(current_scores)
        # convert the size of all input to beam_size

        hidden = decoder_init_hidden.unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, batch_size*beam_size, decoder_init_hidden.size(2))  # (2,B*beam_size,C)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, encoder_output.size(1), encoder_output.size(2))  # (B*beam_size,T,C)
        temp_seg = temp_seg.unsqueeze(1).repeat(1, beam_size, 1).view(-1, 2)  # (B*beam_size, 2)
        encoding_mask = encoding_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, encoding_mask.size(1), 1)  #(B*beam_size,T,1)

        # next_input_word = Variable(torch.LongTensor([BOS_ID]*batch_size*beam_size).cuda())  # batch*beam_size
        next_input_word = Variable(torch.LongTensor([BOS_ID]*batch_size*beam_size))  # B*beam_size [0,0,.....0]

        # forward beam_search
        for step in range(1, self.max_cap_length + 1):  # the first word is said to be <bos>

            input_word_embedding = self.embedding(next_input_word).unsqueeze(1)  # (B*beam_size, 1, 512)

            context = self.attention_module(encoder_output, hidden_transpose(hidden), temp_seg, encoding_mask)  # (B*beam_size, 1536)
            inputs = torch.cat([input_word_embedding, context.unsqueeze(1)], dim=2)  # (B*beam_size, 1, 2048)
            rnn_output, hidden = self.rnn_cell(inputs, hidden)  # (B*beam_size, 1, 512)   (2, B*beam_size, 512)
            output = F.log_softmax(self.output_layer(rnn_output.squeeze(1)), dim=1)  # (B*beam_size, 500)

            output_scores = output.view(batch_size, beam_size, -1) # (B, beam_size, 500)
            output_scores = output_scores + current_scores.unsqueeze(2)  # (B,beam_size,500)  为什么要加上上一步的得分
            current_scores, out_candidate = output_scores.view(batch_size, -1).topk(beam_size, dim=1)  # (B,beam_size)

            next_input_word = (out_candidate % self.vocab_size).view(-1)  # (B*beam_size) 将得到的单词索引展开
            parents = (out_candidate / self.vocab_size).view(batch_size, beam_size)  # (B, beam_size)  [[0,0,0],....,[0,0,0]]   这步操作是用来指示得到的candidate坐标属于哪一个beam_size用于接下来索引对应的隐状态
            hidden_gather_idx = parents.view(1, batch_size, beam_size, 1).expand(hidden.size(0), batch_size, beam_size, hidden.size(2))  # (2, B, beam_size, 512)
            hidden = hidden.view(-1, batch_size, beam_size, hidden.size(2)).gather(dim=2, index=hidden_gather_idx).view(-1, batch_size*beam_size, hidden.size(2))  # (2, B*beam_size, 512)

            out_pred_target_list.append(next_input_word.view(batch_size, beam_size))  # (batch_size,beam_size)
            out_pred_parent_list.append(parents)

            end_mask = next_input_word.data.eq(EOS_ID)
            tmp = end_mask.nonzero().dim()
            if end_mask.nonzero().dim() > 0:
                stored_scores = current_scores.clone()  # (B,3)
                # current_scores.data.masked_fill_(end_mask, -float('inf'))
                # stored_scores.data.masked_fill_(end_mask == False, -float('inf'))

                current_scores.view(-1).data.masked_fill_(end_mask, -float('inf'))
                stored_scores.view(-1).data.masked_fill_(end_mask == False, -float('inf'))
                candidate_score_dict[step] = stored_scores

        # back track 先获取最后一步的最大啊结果再倒过来反推前面的
        final_pred = list()
        # seq_length = Variable(torch.LongTensor(batch_size).zero_().cuda()) + 1
        seq_length = Variable(torch.LongTensor(batch_size).zero_()) + 1  # [1,1,1,1,1,1,1,1,1,1,1,1]
        max_score, current_idx = current_scores.max(1)  # (B)
        current_idx = current_idx.unsqueeze(1) # (B,1)
        # final_pred.append(Variable(torch.zeros(batch_size, 1).long().cuda()) + EOS_ID)
        final_pred.append(Variable(torch.zeros(batch_size, 1).long()) + EOS_ID)  # [1],[1],[1]....[1]
        for step in range(self.max_cap_length, 0, -1):
            if step in candidate_score_dict:  # we find end index
                max_score, true_idx = torch.cat([candidate_score_dict[step], max_score.unsqueeze(1)], dim=1).max(1)  # (B,3) + (B,1)--> (B)     true_idx:[3,3,3,3,3,3,3,3,3,3,3,3]
                current_idx[true_idx != beam_size] = true_idx[true_idx != beam_size]  # [0,0,0,....0]
                seq_length[true_idx != beam_size] = 0
            final_pred.append(out_pred_target_list[step].gather(dim=1, index=current_idx))  # batch, 1
            current_idx = out_pred_parent_list[step].gather(dim=1, index=current_idx)  # batch, 1  不断更新beam_size的索引
            seq_length = seq_length + 1
        final_pred.append(out_pred_target_list[0].gather(dim=1, index=current_idx))
        seq_length = seq_length + 1
        final_pred = torch.cat(final_pred[::-1], dim=1)  # (B,seq_len+1)  倒序拼接 [0,x,x,x,....,1]

        # caption_mask = Variable(torch.LongTensor(batch_size, self.max_cap_length + 2).zero_().cuda())
        caption_mask = Variable(torch.LongTensor(batch_size, self.max_cap_length + 2).zero_())   # (B,max_cap_length+2)  全0
        # caption_mask_helper = Variable(torch.LongTensor(range(self.max_cap_length + 2)).unsqueeze(0).repeat(batch_size, 1).cuda())
        caption_mask_helper = Variable(torch.LongTensor(range(self.max_cap_length + 2)).unsqueeze(0).repeat(batch_size, 1)) # # (B,max_cap_length+2) [[0,1,....max_len-1],[0,1,....max_len-1],...]
        caption_mask[caption_mask_helper < seq_length.unsqueeze(1)] = 1

        return None, final_pred.detach(), seq_length.detach(), caption_mask.detach()
