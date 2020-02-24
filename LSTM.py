import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size, dropout,layers, isDense):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Initialize LSTM Model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers)

        self.dropout = nn.Dropout(dropout)

        # # Define the fully connected layer
        # self.fc = nn.Linear(self.hidden_dim, 512)

        self.dense = nn.Linear(hidden_dim,hidden_dim)
        self.isDense = isDense

        # Initialize Linear Layer
        self.linear = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):

        # Get LSTM Output
        lstm_out, _ = self.lstm(sentence)
        lstm_out = self.dropout(lstm_out)

        # fc_out = self.fc(lstm_out)

        # Get Linear Layer Output
        if self.isDense:
            dense_out = self.dense(lstm_out.view(len(sentence), -1))
            raw_output = self.linear(dense_out)
        else:
            raw_output = self.linear(lstm_out.view(len(sentence), -1))


        # Normalizes between 0 and 1
        output = F.softmax(raw_output, dim=1)

        # This gets the very last output of the lstm. Since LSTM has an output for each word.
        output_length = list(output.size())[0]-1
        output = output[output_length-1:output_length,:]
        # output3 = output.transpose(0,1)

        return output