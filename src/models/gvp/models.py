import math
import numpy as np
import torch
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
import torch.utils.checkpoint as checkpoint
import sys
from torch_geometric.nn import aggr

class SSEmbGNN(torch.nn.Module):
    '''
    GVP-GNN for structure-conditioned autoregressive
    protein design as described in manuscript.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of
    shape [n_nodes, 20].

    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.

    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 num_layers=4, drop_rate=0.0, vector_gate=True):

        super(SSEmbGNN, self).__init__()

        # Get correct dimensions
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate),
            LayerNorm(edge_h_dim)
        )

        # Encode
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, vector_gate=vector_gate)
            for _ in range(num_layers))

        self.W_S = nn.Embedding(21, 21)

        self.W_M = nn.Sequential(
            nn.Linear(768, node_h_dim[0]),
            )
        
        self.W_decoder_in = nn.Sequential(
            nn.Linear(node_h_dim[0]*2, node_h_dim[0]*2),
            nn.LayerNorm(node_h_dim[0]*2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(node_h_dim[0]*2, node_h_dim[0]),
            nn.LayerNorm(node_h_dim[0]),
        )

        # Decode
        node_h_dim = (node_h_dim[0], node_h_dim[1])
        edge_h_dim = (edge_h_dim[0] + 21, edge_h_dim[1])

        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim,
                             drop_rate=drop_rate, vector_gate=vector_gate)
            for _ in range(num_layers))

        # Out
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None), vector_gate=vector_gate)

    def forward(self, h_V, edge_index, h_E, msa_emb, seq, get_emb=False):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        # Run through GVP to get correct hidden dimensions
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # Message passing
        # Encoding
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        # Add sequence info
        h_S = self.W_S(seq)
        h_S = h_S[edge_index[0]]
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        h_M = self.W_M(msa_emb)
        h_V = (self.W_decoder_in(torch.cat([h_V[0], h_M], dim=-1)), h_V[1])
        
        # Decoding
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Out
        if get_emb == True:
            return h_V[0]
        else:
            logits = self.W_out(h_V)
            return logits
