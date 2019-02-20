import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):

    def __init__(self, in_dim, out_dim, n_layers, hidden_dims, averaging=False):
        if n_layers > len(hidden_dims):
            raise Exception('number of layers doesn\'t match the ones given')

        super(NeuralNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.averaging = averaging
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dims[0])])
        self.layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(0, n_layers-1)])
        self.layers.append(nn.Linear(hidden_dims[n_layers-1], out_dim))

    #This forces an iid uniform initialization, the default initialization for torch has a special structure that depends on the size of the layers
    def init_weights(self, cte=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-cte,cte)
                if m.bias is not None:
                    m.bias.data.uniform_(-cte,cte)

    def freeze_params(self, layer_idx):
        for idx, param in enumerate(self.parameters()):
            if idx in layer_idx:
                param.requires_grad = False

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            l_x = layer(x)

        if l>0 and self.averaging:
            l_x = torch.div(l_x, self.hidden_dims[l-1])

        #ReLU is forced here, the class can be improved to flexibilize this hyperparameter
        if l < self.n_layers:
            x = F.relu(l_x)

        return l_x
