import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class LightGCN(nn.Module):

    def __init__(self, device, n_layers, n_users, n_items, interaction_matrix):
        super(LightGCN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.user_count = n_users
        self.item_count = n_items
        self.interaction_matrix = interaction_matrix
        self.A_adj_matrix = self._get_a_adj_matrix()

    def _get_a_adj_matrix(self):
        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.transpose()
        n = self.user_count + self.item_count
        rows = np.concatenate([inter_matrix.row, inter_matrix_t.row + self.user_count])
        cols = np.concatenate([inter_matrix.col + self.user_count, inter_matrix_t.col])
        data = np.ones(len(rows), dtype=float)
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        A = A.tocsr()
        A.data = np.ones_like(A.data)
        A = A.tocoo()
        sum_list = np.array(A.sum(axis=1)).flatten()
        diag = sum_list + 1e-07
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        A_adj = D * A * D
        A_adj = sp.coo_matrix(A_adj)
        row = A_adj.row
        col = A_adj.col
        index = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse.FloatTensor(index, data, torch.Size(A_adj.shape))
        return A_sparse

    def forward(self, in_embs):
        result = [in_embs]
        for i in range(self.n_layers):
            in_embs = torch.sparse.mm(self.A_adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))
            result.append(in_embs)
        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)
        return result
