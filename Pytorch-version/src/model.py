import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SyntaxEmbeding(nn.Module):

    def __init__(self, syntax_len, syntax_depth, syntax_emd_size):
        super(SyntaxEmbeding, self).__init__()
        self.embedding = nn.Embedding(syntax_len + 1, syntax_emd_size)
        self.position_embedding = nn.Embedding(syntax_depth, syntax_emd_size).weight
        # self.position_embedding.requires_grad = False

    def forward(self, syntax):
        syntax_embedding = self.embedding(syntax)
        output = torch.sum(torch.mul(syntax_embedding, self.position_embedding), dim=2)
        return output


class TextCNN(nn.Module):
    def __init__(self, dim_in, dim_out, embeding_size, filter_list, dropout=0.25):
        super(TextCNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(dim_in, dim_out, (i, embeding_size)) for i in filter_list])

    def conv_and_pool(self, inputs, conv):
        inputs = F.relu(conv(self.dropout(inputs))).squeeze(3)
        inputs = F.max_pool1d(inputs, inputs.size(2)).squeeze(2)
        return inputs

    def forward(self, inputs):
        outs = [self.conv_and_pool(inputs, conv) for conv in self.conv_list]
        outs = torch.cat(outs, 1)
        return outs


class SyntaxTextCNN(nn.Module):
    def __init__(self, text_dim_out, syntax_dim_out, text_emd_size, syntax_emd_size,
                 char_len, syntax_len, syntax_depth, filter_list, cls_num, dropout=0.25):
        super(SyntaxTextCNN, self).__init__()
        self.text_embedding = nn.Embedding(char_len, text_emd_size)
        self.syntax_embedding = SyntaxEmbeding(syntax_len, syntax_depth, syntax_emd_size)
        self.text_cnn = TextCNN(1, text_dim_out, text_emd_size, filter_list)
        self.syntax_cnn = TextCNN(1, syntax_dim_out, syntax_emd_size, filter_list)
        self.dropout = nn.AlphaDropout(dropout)
        self.fc1 = nn.Linear((text_dim_out + syntax_dim_out) * len(filter_list), cls_num)

    def forward(self, text, syntax):
        text_output = self.text_embedding(text)
        text_output = text_output.unsqueeze(1)
        text_output = self.dropout(self.text_cnn(text_output))

        syntax_output = self.syntax_embedding(syntax)
        syntax_output = syntax_output.unsqueeze(1)
        syntax_output = self.dropout(self.syntax_cnn(syntax_output))

        output = torch.cat([text_output, syntax_output], 1)
        output = self.fc1(output)
        return output


def get_position_embedding(syntax_depth, syntax_emd_size):
    position_embedding = torch.tensor(np.array([
        [pos / np.power(10000, 2.0 * (j // 2) / syntax_emd_size) for j in range(syntax_emd_size)]
        for pos in range(syntax_depth)]), dtype=torch.float)
    # 偶数列使用sin，奇数列使用cos
    position_embedding[:, 0::2] = torch.sin(position_embedding[:, 0::2])
    position_embedding[:, 1::2] = torch.cos(position_embedding[:, 1::2])
    pad_row = torch.zeros([1, syntax_emd_size])
    position_embedding = torch.cat((pad_row, position_embedding))
    return position_embedding[:-1]


if __name__ == '__main__':
    embeding = torch.rand(10, 1, 50, 300)
    syntax = torch.rand(10, 1, 50, 300)
    input = torch.cat((embeding, syntax), dim=1)
    print("==================")
    tc = SyntaxTextCNN(2, 5, 300, 60, 10, 10, 150, [3, 4, 5], 10)
    print(tc)
    for name, p in tc.named_parameters():
        # if p.require_grad:
        print(name)
    # print(tc(input).size())

