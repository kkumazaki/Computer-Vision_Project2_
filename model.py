import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''     
        super(DecoderRNN, self).__init__()

        # initialize hidden size
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)

        # define the final, fully-connected output layer
        self.linear = nn.Linear(hidden_size, vocab_size)


    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        embeds = self.embed(captions[:, :-1])
        embeds = torch.cat((features.unsqueeze(1), embeds), dim=1)

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        self.hidden = self.init_hidden(features.shape[0])
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) 
        hidden = self.init_hidden(features.shape[0])
        lstm_out, self.hidden = self.lstm(embeds, hidden)      

        # output
        output = self.linear(lstm_out)

        # another way
        #hidden, _ = self.lstm(embeds)
        #output = self.linear(hidden)

        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Initialize in sample
        hidden = self.init_hidden(inputs.shape[0])
        samples = []

        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            output = self.linear(lstm_out)

            # another way
            #hidden, states = self.lstm(inputs, states)
            #output = self.linear(hidden)
            
            output = output.squeeze(1)
            output = output.argmax(dim=1)
            samples.append(output.item())
            inputs = self.embed(output.unsqueeze(0))

        return samples