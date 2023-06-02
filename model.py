
import torch
from torch import nn
class Advanced_feature_extractor(nn.Module):

    def __init__(self):
        super(Advanced_feature_extractor, self).__init__()
        self.embedding = nn.Embedding(21, 20)

        self.conv1 = nn.Conv1d(30, 1200, kernel_size=(510,), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)

        self.conv2 = nn.Conv1d(30, 900, kernel_size=(510,), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)

        self.Flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2100, 1000)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        self.linear2 = nn.Linear(1000, 2)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        # self.linear3=nn.Linear(100,2)
        self.BatchNorm1 = nn.LazyBatchNorm1d()

        self.BatchNorm2 = nn.LazyBatchNorm1d()
        self.dropout = nn.Dropout(0.3)  # best=0.3

    def forward(self, input):
        # food_conv1 = self.maxpool1(self.dropout(self.BatchNorm(self.relu(self.food_conv1(input)))).squeeze(3))
        conv1 = self.conv1(input)

        conv1 = self.relu(conv1)
        conv1 = self.BatchNorm1(conv1)
        conv1 = self.dropout(conv1)

        #

        conv2 = self.conv2(input)
        conv2 = self.relu(conv2)
        conv2 = self.BatchNorm2(conv2)
        conv2 = self.dropout(conv2)

        all = torch.cat([conv1, conv2], 1).squeeze(2)

        all = self.linear2(self.relu(self.linear1(all)))

        return all


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=15):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).reshape((-1, 1))
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).reshape((1, -1)) / d_model)
        # sin and cos position encoding

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class PCTE(nn.Module):
    def __init__(self, hidden_dim=510,
                 dim_feedforward=1024, num_head=10, num_layers=3, dropout=0.3, max_len=15, activation: str = "relu"):
        super(PCTE, self).__init__()

        self.embeddings = nn.Embedding(21, 516)
        self.conv1 = nn.Conv1d(15, 15, kernel_size=(3,), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)
        self.conv2 = nn.Conv1d(15, 15, kernel_size=(5,), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)
        self.position_embedding = PositionalEncoding(hidden_dim, dropout, max_len)

        encoder_layer_one = nn.TransformerEncoderLayer(hidden_dim, 10, dim_feedforward, dropout, activation=activation)
        self.transformer_one = nn.TransformerEncoder(encoder_layer_one, num_layers)

        encoder_layer_two = nn.TransformerEncoderLayer(hidden_dim, 5, dim_feedforward, dropout, activation=activation)
        self.transformer_two = nn.TransformerEncoder(encoder_layer_two, num_layers)

        self.output = nn.Linear(hidden_dim, 2)
        self.output2 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.BatchNorm1 = nn.LazyBatchNorm1d()
        self.BatchNorm2 = nn.LazyBatchNorm1d()
        self.dropout = nn.Dropout1d(dropout)
        self.afe = Advanced_feature_extractor()

    def forward(self, inputs):
        attention_mask = self.get_key_padding_mask(tokens=inputs)

        hidden_states = self.embeddings(inputs)
        hidden_states = self.BatchNorm1(self.relu(self.conv1(hidden_states)))
        hidden_states = self.BatchNorm2(self.relu(self.conv2(hidden_states)))  #Primary features

        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.transpose(hidden_states, 0, 1)
        hidden_states = self.position_embedding(hidden_states)

        attention_mask = attention_mask.cuda()
        hidden_states_one = self.transformer_one(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states_two = self.transformer_two(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states_one = torch.transpose(hidden_states_one, 0, 1)
        hidden_states_two = torch.transpose(hidden_states_two, 0, 1)

        tf = torch.cat([hidden_states_one, hidden_states_two], 1)

        # time series feature

        return self.afe(tf)

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.ones(tokens.size())
        key_padding_mask[tokens == 20] = 0
        inv_key_padding_mask = (1 - key_padding_mask)

        return inv_key_padding_mask.to(torch.bool)