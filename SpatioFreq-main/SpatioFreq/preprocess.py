import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from sklearn.decomposition import PCA
def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    #sc.pp.filter_genes(adata, min_cells=1)
    #sc.pp.filter_genes(adata_sc, min_cells=1)
    
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

# def construct_interaction(adata, n_neighbors=3):
#     """Constructing spot-to-spot interactive graph"""
#     position = adata.obsm['spatial']
    
#     # calculate distance matrix
#     distance_matrix = ot.dist(position, position, metric='euclidean')
#     n_spot = distance_matrix.shape[0]
    
#     adata.obsm['distance_matrix'] = distance_matrix
    
#     # find k-nearest neighbors
#     interaction = np.zeros([n_spot, n_spot])  
#     for i in range(n_spot):
#         vec = distance_matrix[i, :]
#         distance = vec.argsort()
#         for t in range(1, n_neighbors + 1):
#             y = distance[t]
#             interaction[i, y] = 1
         
#     adata.obsm['graph_neigh'] = interaction
    
#     #transform adj to symmetrical adj
#     adj = interaction
#     adj = adj + adj.T
#     adj = np.where(adj>1, 1, adj)
    
#     adata.obsm['adj'] = adj


import numpy as np
import ot

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph based on spatial distances"""
    
    # 提取空间坐标，确保为 NumPy 数组
    position = np.array(adata.obsm['spatial'])
    
    # 计算距离矩阵（欧几里得距离）
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    # 将计算出的距离矩阵保存到 adata
    adata.obsm['distance_matrix'] = distance_matrix
    
    # 创建一个 n x n 的邻接矩阵，初始化为 0
    interaction = np.zeros([n_spot, n_spot])
    
    # 为每个点找到 k 最近邻
    for i in range(n_spot):
        vec = distance_matrix[i, :]  # 获取当前点与其他点的距离
        distance = vec.argsort()  # 按照距离排序，返回排序后的索引
        
        # 只保留最近的 n_neighbors 个点（排除第一个点自己）
        for t in range(1, n_neighbors + 1):
            y = distance[t]  # 最近的点索引
            interaction[i, y] = 1  # 设定邻接矩阵为 1
    
    # 将邻接矩阵添加到 adata 中
    adata.obsm['graph_neigh'] = interaction
    
    # 将邻接矩阵转换为对称矩阵（无向图）
    adj = interaction + interaction.T  # 对称化
    adj = np.where(adj > 1, 1, adj)  # 确保邻接矩阵中的值为 0 或 1
    
    # 将最终的邻接矩阵添加到 adata 中
    adata.obsm['adj'] = adj
    
def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    print('Graph constructed!')   

def preprocess(adata,n_top_genes):
    # # 提取特征矩阵
    # features = adata.X
    
    # # 执行 PCA 降维
    # pca = PCA(n_components=5723)
    # features_pca = pca.fit_transform(features)
    
    # # 将 PCA 降维后的结果替换原始数据的特征矩阵
    # adata.X = features_pca
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
