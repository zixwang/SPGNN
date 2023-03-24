import logging
import random
import pandas as pd
import numpy as np
import torch
import string
import scipy.sparse as sp
from model import *
from utils import *
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import NormalizeFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_num_threads(1)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d %b %Y, %H:%M:%S',
                    filename='log/SGC-normalize.log',
                    filemode='w')

match_table = pd.read_csv("dataset/mirna_lncrna_interaction.csv")
unique_lnc = list(set(match_table['lncrna']))
unique_mi = list(set(match_table['mirna']))

lnc_seq = []
mi_seq = []

for i in unique_lnc:
    seq = match_table[match_table['lncrna'] == i]["lncrna_seq"]
    seq = list(seq)
    seq = seq[0]
    seq = seq.translate(str.maketrans('', '', string.punctuation))
    lnc_seq.append(seq)

for i in unique_mi:
    seq = match_table[match_table['mirna'] == i]["mirna_seq"]
    seq = list(seq)
    seq = seq[0]
    seq = seq.replace('.', '')
    if ',' in seq:
        seq = seq.split(',')
        seq = seq[0]

    mi_seq.append(seq)


lnc_seq_mers = []
mi_seq_mers = []

for i in lnc_seq:
    lnc_seq_mers.append(k_mers(3, i))

for i in mi_seq:
    mi_seq_mers.append(k_mers(3, i))


all_mers = lnc_seq_mers + mi_seq_mers
all_name = unique_lnc + unique_mi
pretrain_model = train_doc2vec_model(all_mers, all_name)
vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)

# Create Graph Embeddings
graph_table = pd.read_csv("dataset/index_value.csv")
graph_label = list(graph_table["rna"])
graph_embedding = np.zeros((len(graph_label), 100))
for node, vec in vectors.items():
    position = graph_label.index(node)
    graph_embedding[position] = vec
x_embedding = torch.tensor(graph_embedding).float()

# Construct DGL Graph
node_table = pd.read_csv("dataset/node_link.csv")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    y_true = data.edge_label.cpu().numpy()
    y_pred = out.cpu().numpy()

    roc_auc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    ndcg = NDCG(y_true, y_pred)

    return roc_auc, avg_precision, ndcg


best_epoches = []
auc_scores = []
ap_scores = []
ndcg_scores = []


# Construct 5-fold cross validation pipeline
for train_index, val_index in kfold.split(node_table):
    train_set = node_table.iloc[train_index]
    val_set = node_table.iloc[val_index]

    # Construct train Graph edges
    u_float_train = list(train_set['node1'])
    u_train = [int(x) for x in u_float_train]
    v_float_train = list(train_set['node2'])
    v_train = [int(y) for y in v_float_train]
    u_undirected_train = [x for pair in zip(
        u_train, v_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
    v_undirected_train = [x for pair in zip(
        v_train, u_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
    u_undirected_train = torch.tensor(u_undirected_train)
    v_undirected_train = torch.tensor(v_undirected_train)

    # Construct validation Graph edges
    u_float_val = list(val_set['node1'])
    u_val = [int(x) for x in u_float_val]
    v_float_val = list(val_set['node2'])
    v_val = [int(y) for y in v_float_val]
    u_undirected_val = [x for pair in zip(
        u_val, v_val) for x in pair] + u_val[len(v_val):] + v_val[len(u_val):]
    v_undirected_val = [x for pair in zip(
        v_val, u_val) for x in pair] + u_val[len(v_val):] + v_val[len(u_val):]
    u_undirected_val = torch.tensor(u_undirected_val)
    v_undirected_val = torch.tensor(v_undirected_val)

    # Construct train_data and val_data
    edge_index = torch.stack(
        [u_undirected_train, v_undirected_train], dim=0)
    edge_train_index = torch.stack(
        [torch.tensor(u_train), torch.tensor(v_train)], dim=0)
    edge_val_index = torch.stack(
        [torch.tensor(u_val), torch.tensor(v_val)], dim=0)

    train_data = Data(x=x_embedding, edge_index=edge_index, edge_label=torch.ones(
        len(u_train)), edge_label_index=edge_train_index).to(device)

    val_data = Data(x=x_embedding, edge_index=edge_index, edge_label=torch.ones(
        len(u_val)), edge_label_index=edge_val_index).to(device)

    neg_edge_index = negative_sampling(
        edge_index=val_data.edge_index, num_nodes=val_data.num_nodes,
        num_neg_samples=val_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [val_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        val_data.edge_label,
        val_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    val_data = Data(x=x_embedding, edge_index=edge_index,
                    edge_label=edge_label, edge_label_index=edge_label_index).to(device)

    transform = NormalizeFeatures()
    train_data = transform(train_data)
    val_data = transform(val_data)
    
    # Construct model and optimizer
    # Net: GCNNet SAGENet GATNet SGCNet FiLMNet GATv2Net
    model = SGCNet(train_data.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train and Validation
    results = []
    best_val_auc = 0
    for epoch in range(1, 501):
        loss = train()
        val_auc, val_ap, val_ndcg = test(val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')
        results.append([epoch, val_auc, val_ap, val_ndcg])

    best_result = max(results, key=lambda x: x[1])
    log_and_print('Best result: Epoch: {}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}'.format(
        *best_result))
    best_epoches.append(best_result[0])
    auc_scores.append(best_result[1])
    ap_scores.append(best_result[2])
    ndcg_scores.append(best_result[3])

log_and_print("-----------------------Final Results-----------------------")
log_and_print('AUC scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(auc_scores), np.std(auc_scores)))
log_and_print('AP scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(ap_scores), np.std(ap_scores)))
log_and_print('NDCG scores: mean {:.3f}, std {:.3f}'.format(
    np.mean(ndcg_scores), np.std(ndcg_scores)))
log_and_print("-----------------------------------------------------------")
