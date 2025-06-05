import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First GAT layer: double the output channels and use 2 attention heads
        self.conv1 = GATConv(in_channels, 2 * out_channels, heads=2, concat=True, dropout=0.2)
        # Second GAT layer: same logic again to increase model depth
        self.conv2 = GATConv(2 * out_channels * 2, 2 * out_channels, heads=2, concat=True, dropout=0.2)
        # Output mean and log-variance layers for the latent distribution
        self.conv_mu = GATConv(2 * out_channels * 2, out_channels, heads=1, concat=False, dropout=0.2)
        self.conv_logstd = GATConv(2 * out_channels * 2, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd
