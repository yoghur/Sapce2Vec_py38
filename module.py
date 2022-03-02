import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import torch.utils.data
import math
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
class LayerNorm(nn.Module):
    """
    layer norm
    """
    def __init__(self,feature_dim,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma",self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta",self.beta)
        self.eps = eps

    def forward(self,x):
        #x:[batch_size,embed_dim]
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps)+self.beta

def get_activation_function(activation,context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str+' '+activation))

class SingleFeedForwardNN(nn.Module):
    """
    single layer fully connected feed forward neural network
    """
    def __init__(self,input_dim,
                    output_dim,
                    dropout_rate=None,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=""):
        super(SingleFeedForwardNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None
        
        self.act = get_activation_function(activation,context_str)

        if use_layernormalize:
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None
        
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False
        
        self.linear = nn.Linear(self.input_dim,self.output_dim)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self,input_tensor):
        assert input_tensor.size()[-1] == self.input_dim
        #Linear layer
        
        input_tensor = input_tensor.to(device)
        # print(device,input_tensor.is_cuda)
        output = self.linear(input_tensor)
        #non-linerity
        output = self.act(output)

        if self.dropout is not None:
            output = self.dropout(output)
        
        if self.skip_connection:
            output = output + input_tensor
        
        if self.layernorm is not None:
            output = self.layernorm(output)
        
        return output

class MultiLayerFeedForwardNN(nn.Module):
    """
    N fully connected feed forward NN
    """
    def __init__(self,input_dim,
                    output_dim,
                    num_hidden_layers=0,
                    dropout_rate=None,
                    hidden_dim=-1,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=None):
        super(MultiLayerFeedForwardNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))
            for i in range(self.num_hidden_layers-1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
    
    def forward(self,input_tensor):
        assert input_tensor.size()[-1] == self.input_dim
        input_tensor = input_tensor.to(device)
        # print(input_tensor.is_cuda)
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)
        return output

class ResLayer(nn.Module):
    def __init__(self,linear_size):
        super(ResLayer,self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size,self.l_size)
        self.w2 = nn.Linear(self.l_size,self.l_size)

    def forward(self,x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x+y
        return out

class FCNet(nn.Module):
    def __init__(self,num_inputs,num_filts,num_hidden_layers):
        super(FCNet,self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.feats = nn.Sequential()
        self.feats.add_module("ln_1",nn.Linear(num_inputs,num_filts))
        self.feats.add_module("relu_1",nn.ReLU(inplace=True))
        for i in range(num_hidden_layers):
            self.feats.add_module("resnet_{}".format(i+1),ResLayer(num_filts))

    def forward(self,x):
        loc_emb = self.feats(x)
        return loc_emb
