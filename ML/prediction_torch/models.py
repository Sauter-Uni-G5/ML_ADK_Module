import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from abc import ABC

class BaseRNNModule(nn.Module, ABC):
    '''
    Classe Base para módlos de Redes Neurais para predição de séries temporarais. Usa a biblioteca PyTorch.
    
    Args:
        module (nn.Module): Módulo de RNN do PyTorch (nn.RNN, nn.LSTM, nn.GRU).
        input_size (int): Tamanho da entrada.
        hidden_size (int): Tamanho do estado oculto.
        num_layers (int): Número de camadas na RNN.
        output_size (int): Tamanho da saída (Horizon)
        bidirectional (bool): Se a RNN é bidirecional ou não (o fluxo inverso fecha os gaps de informação)
    '''
    def __init__(self, module, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.module = module(input_size=self.input_size, hidden_size= self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=bidirectional)
        self._init_parameters()
    
    def _init_parameters(self):
        for name, param in self.module.named_parameters():
            print(f'Initializing {name}')
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain("tanh"))
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)

    def forward(self, x):
        output, h_n = self.module(x)
        output = output[:,-1, :]
        out = self.linear(output) # Transformação linear para ajustar a dimensão de saída
        return out
    
    def predict_window(model, X_init, steps):
        model.eval()
        preds = []
        X_seq = X_init.clone()

        with torch.no_grad():
            for _ in range(steps):
                out = model(X_seq) 
                preds.append(out.item())

                new_val = out.unsqueeze(0).unsqueeze(0)
                X_seq = torch.cat([X_seq[:,1:,:], new_val], dim=1)

        return preds

class RNN_model(BaseRNNModule):
    def __init__(self, input_size, hidden_size, num_layers, ouput_size, bidirectional=False):
        super().__init__(module=nn.RNN, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,  output_size=ouput_size, bidirectional=bidirectional)
        direction_factor = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * direction_factor, self.output_size)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

class LSTM_model(BaseRNNModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super().__init__(module=nn.LSTM, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, bidirectional=bidirectional)
        direction_factor = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * direction_factor, self.output_size)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

class GRU_model(BaseRNNModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super().__init__(module=nn.GRU, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, bidirectional=bidirectional)
        direction_factor = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * direction_factor, self.output_size)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)