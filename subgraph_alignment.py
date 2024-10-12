import numpy as np
import scipy.sparse as sp
from embedding import createWlEmbedding, createWlEmbedding_new


def generate_hnodes(h_adj):
    h_adj = h_adj.tocoo()
    h_index = [[] for i in range(h_adj.shape[0])]
    for i, j in zip(h_adj.row, h_adj.col):
        h_index[i].append(j)
    return h_index


def generate_hadj(adj, h):
    adj_h = sp.eye(adj.shape[0])
    adj_tot = sp.eye(adj.shape[0])
    for i in range(h):
        adj_h = adj_h * adj
        adj_tot = adj_tot + adj_h
    return adj_tot


# Generate h_nodes and their height
def generate_h_nodes_n_dict(adj, h):
    adj_h = sp.eye(adj.shape[0])
    M = [{i: 0} for i in range(adj.shape[0])]
    h_index = [[i] for i in range(adj.shape[0])]
    for _ in range(h):
        adj_h = sp.coo_matrix(adj_h * adj)

        for i, j in zip(adj_h.row, adj_h.col):
            if j in M[i]:
                continue
            else:
                M[i][j] = _ + 1
                h_index[i].append(j)
    return M, h_index


def generate_index(adj, h):
    h_adj = generate_hadj(adj, h)
    subgraph_index = generate_hnodes(h_adj)
    return subgraph_index


def generate_subembeddings(attr, adj, subgraph_index, h):
    embedding = []
    for i in range(adj.shape[0]):
        root_feature = attr[i, :]
        feature = attr[subgraph_index[i]]
        feature = feature - np.tile(root_feature, (len(subgraph_index[i]), 1))
        adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]]
        embedding.append(createWlEmbedding(feature, adj_i, h))
    return np.concatenate(embedding, axis=0)


def subgraph_embeddings(attr, adj, h):
    subgraph_index = generate_index(adj, h)
    embedding = generate_subembeddings(attr, adj, subgraph_index, h)
    return embedding


def generate_subembeddings1(attr, adj, subgraph_index, h):
    embedding = []
    for i in range(adj.shape[0]):
        root_feature = attr[i, :]
        feature = attr[subgraph_index[i]]
        # feature = feature - np.tile(root_feature, (len(subgraph_index[i]), 1))
        adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]]
        embedding.append(createWlEmbedding_new(feature, adj_i, h))
    return np.concatenate(embedding, axis=0)


def subgraph_embeddings1(attr, adj, h):
    subgraph_index = generate_index(adj, h)
    embedding = generate_subembeddings1(attr, adj, subgraph_index, h)
    return embedding

# Generate nodes index via dict


def subgraph_embeddings2(attr, adj, h):
    M, h_index = generate_h_nodes_n_dict(adj, h)
    embedding = generate_subembeddings(attr, adj, h_index, h)
    return embedding, M
