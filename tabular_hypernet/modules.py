from dotenv import load_dotenv
load_dotenv()

import os

import torch
import numpy as np
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


class SimpleNetwork(torch.nn.Module):
    def __init__(self, inp_size, layers=[100], outputs=10):
        super().__init__()
        self.layers = []
        
        self.inp = torch.nn.Linear(inp_size, layers[0])
        self.output = torch.nn.Linear(layers[0], outputs)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        out = x
        out = self.inp(out)
        out = self.relu(out)
        out = self.output(out)
        return out
    
class DropoutNetwork(torch.nn.Module):
    def __init__(self, inp_size, layers=[100]):
        super().__init__()
        self.layers = []
        
        self.inp = torch.nn.Linear(inp_size, layers[0])
        self.output = torch.nn.Linear(layers[0], 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(1-700/784)
    
    def forward(self, x):
        out = x
        out = self.dropout(out)
        out = self.inp(out)
        out = self.relu(out)
        out = self.output(out)
        return out

    
class InsertableNet(torch.nn.Module):
    def __init__(self, weights, shape=[(784, 10), (10, 10)]):
        super().__init__()
        self.layers = []
        self._offset = 0
        
        for layer in shape:
            _w_size = layer[0]*layer[1]
            _b_size = layer[1]
            
            _l = (weights[self._offset:self._offset+_w_size].reshape((layer[1], layer[0])),
                  weights[self._offset+_w_size:self._offset+_w_size+_b_size])
            self._offset += _w_size+_b_size

            self.layers.append(_l)
    
    def forward(self, data):
        out = data
        for layer in self.layers[:-1]:
            out = F.linear(out, layer[0], layer[1])
            out = F.relu(out)

        return F.linear(out, self.layers[-1][0], self.layers[-1][1])
    
    
class MaskedNetwork(SimpleNetwork):
    def __init__(self, input_size, mask_size, layers=[10]):
        super().__init__(mask_size, layers=layers)
        template = np.zeros(input_size)
        mask = np.random.choice(len(template), mask_size, False)
        template[mask] = 1
        self.mask = torch.from_numpy(template).to(torch.bool)
        
    def forward(self, x):
        data = x[:, self.mask]
        return super().forward(data)
