import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SGConv, FiLMConv, GATv2Conv


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class SGCNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SGConv(in_channels, hidden_channels, K=2)
        self.conv2 = SGConv(hidden_channels, out_channels, K=2)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class FiLMNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = FiLMConv(in_channels, hidden_channels)
        self.conv2 = FiLMConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GATv2Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# ----------------------------------------------------------------------------------------------------------
# class SGC(nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(SGC, self).__init__()
#         self.sgc1 = SGConv(in_feats, h_feats, allow_zero_in_degree=True)
#         self.sgc2 = SGConv(h_feats, h_feats, allow_zero_in_degree=True)

#     def forward(self, g, in_feat):
#         h = self.sgc1(g, in_feat)
#         h = torch.relu(h)
#         h = self.sgc2(g, h)
#         return h


# class GraphSAGE(nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(GraphSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
#         self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         return h


# class GAT(nn.Module):
#     def __init__(self, in_feats, out_feats, num_heads):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(in_feats, out_feats, num_heads,
#                              allow_zero_in_degree=True)
#         self.conv2 = GATConv(out_feats * num_heads,
#                              out_feats, 1, allow_zero_in_degree=True)

#     def forward(self, g, inputs):
#         h = self.conv1(g, inputs)
#         h = h.view(h.shape[0], -1)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         h = h.view(h.shape[0], -1)
#         return h


# class GATv2(nn.Module):
#     def __init__(self, in_feats, out_feats, num_heads):
#         super(GATv2, self).__init__()
#         self.conv1 = GATv2Conv(in_feats, out_feats,
#                                num_heads, allow_zero_in_degree=True)
#         self.conv2 = GATv2Conv(out_feats * num_heads,
#                                out_feats, 1, allow_zero_in_degree=True)

#     def forward(self, g, inputs):
#         h = self.conv1(g, inputs)
#         h = h.view(h.shape[0], -1)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         h = h.view(h.shape[0], -1)
#         return h


# class GCN(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, out_feats, allow_zero_in_degree=True)
#         self.conv2 = GraphConv(out_feats, out_feats, allow_zero_in_degree=True)

#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = torch.relu(h)
#         h = self.conv2(g, h)
#         return h
