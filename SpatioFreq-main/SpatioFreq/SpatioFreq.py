import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse, Encoder_map, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import normalize
import numpy as np
from scipy.fft import fft, ifft


def filter_frequency_signal(freq_signal, method='lowpass', cutoff=0.1, smooth=True, sigma=2, wavelet='db1', level=None):
    """
    过滤频域信号，支持低通滤波和小波变换方法。

    参数:
        freq_signal (array): 输入信号（频域信号）。
        method (str): 滤波方法，'lowpass' 或 'wavelet'。默认 'lowpass'。
        cutoff (float): 低通滤波器的截止频率 (0.0 ~ 0.5)，默认 0.1。仅在 'lowpass' 方法下使用。
        smooth (bool): 是否对信号进行高斯平滑，默认 True。
        sigma (float): 高斯平滑的标准差，默认 2。
        wavelet (str): 小波类型，默认 'db1'，仅在 'wavelet' 方法下使用。
        level (int): 小波变换的分解层数，默认 None，只有在 'wavelet' 方法下才有效。

    返回:
        filtered_signal (array): 经过滤波后的信号。
    """
    if method == 'lowpass':
        # 低通滤波
        fft_coeffs = fft(freq_signal)
        
        # 构造低通滤波器，保留低频部分
        freqs = np.fft.fftfreq(len(freq_signal))
        lowpass_filter = np.abs(freqs) < cutoff  # 保留低于 cutoff 的频率
        
        # 应用低通滤波器
        filtered_fft_coeffs = fft_coeffs * lowpass_filter
        
        filtered_signal = np.real(ifft(filtered_fft_coeffs))

    elif method == 'wavelet':
        # 小波变换
        coeffs = pywt.wavedec(freq_signal, wavelet, level=level)
        threshold_value = np.sqrt(2 * np.log(freq_signal.size))  # 计算阈值
        coeffs[1:] = [pywt.threshold(c, value=threshold_value, mode='soft') for c in coeffs[1:]]  # 软阈值
        filtered_signal = pywt.waverec(coeffs, wavelet)[:len(freq_signal)]  # 重构信号

    else:
        raise ValueError("Unsupported method. Only 'lowpass' and 'wavelet' are supported.")

    # 可选：高斯平滑
    if smooth:
        filtered_signal = gaussian_filter1d(filtered_signal, sigma=sigma)
    
    return filtered_signal


# -------------------- 频域特征提取相关函数 拉普拉斯1--------------------
def get_laplacian_mtx(X, num_neighbors=6, normalization=False):
    """
    计算拉普拉斯矩阵。
    """
    from sklearn.neighbors import kneighbors_graph
    # 邻接矩阵
    adj_mtx = kneighbors_graph(X, n_neighbors=num_neighbors, mode='connectivity', include_self=True)
    adj_mtx = ss.csr_matrix(adj_mtx)

    # 度矩阵
    deg_mtx = np.array(adj_mtx.sum(axis=1)).flatten()
    deg_mtx = np.diagflat(deg_mtx)

    # 拉普拉斯矩阵
    if not normalization:
        lap_mtx = deg_mtx - adj_mtx
    else:
        deg_inv_sqrt = np.diagflat(np.power(deg_mtx.diagonal(), -0.5))
        lap_mtx = np.identity(deg_mtx.shape[0]) - deg_inv_sqrt @ adj_mtx @ deg_inv_sqrt
    return lap_mtx

def obtain_freq_spots(adata, lap_mtx, n_fcs, c=1):
    """
    获取频域特征矩阵。
    """
    if isinstance(adata, torch.Tensor):
        X = adata.cpu().detach().numpy()
    else:
        X = adata if not ss.issparse(adata) else adata.A

    X = normalize(X, norm='max', axis=0)
    X = np.matmul(X, X.T)
    X = normalize(X)

    n_fcs = min(n_fcs, X.shape[0] - 1)

    # 拉普拉斯特征分解
    v0 = [1 / np.sqrt(lap_mtx.shape[0])] * lap_mtx.shape[0]
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx, k=n_fcs, which='SM', v0=v0)
    power = [1 / (1 + c * eigv) for eigv in eigvals]
    eigvecs = np.matmul(eigvecs, np.diag(power))

    freq_mtx = np.matmul(eigvecs.T, X)
    freq_mtx = normalize(freq_mtx[1:, :], norm='l2', axis=0).T
    return freq_mtx, eigvecs.T



def plot_signals_overall(original_signals, filtered_signals, title="Overall Signal Comparison"):
    """
    绘制整体信号，将所有维度的信号求平均后绘制。
    
    参数：
    - original_signals: 原始信号矩阵 (n_samples, n_features)
    - filtered_signals: 过滤后的信号矩阵 (n_samples, n_features)
    - title: 图表标题
    """
    # 计算每个时间点的均值
    avg_original_signal = np.mean(original_signals, axis=1)
    avg_filtered_signal = np.mean(filtered_signals, axis=1)

    # 绘制平均信号
    plt.figure(figsize=(8, 6))
    plt.plot(avg_original_signal, label="Before Filtering", alpha=0.7, color="blue")
    plt.plot(avg_filtered_signal, label="After Filtering", alpha=0.7, color="red")
    
    # 设置标题和标签字体大小
    plt.title(title, fontsize=16)
    plt.xlabel("Samples", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    
    # 设置图例和字体大小
    plt.legend(fontsize=12)
    
    # 不显示网格
    plt.grid(False)
    
    # 不显示边框
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # 显示图表
    plt.show()

from numpy.fft import fft, fftfreq  
def plot_frequency_comparison(original_signal, filtered_signal, fs=1000):
    """
    绘制原始信号和滤波后的信号的频域对比图。
    
    参数：
    - original_signal: 原始信号。
    - filtered_signal: 经过滤波后的信号。
    - fs: 采样频率，默认值为1000。
    """
    # 计算傅里叶变换
    X_fft_original = np.fft.fft(original_signal)
    X_fft_filtered = np.fft.fft(filtered_signal)

    # 计算幅度谱
    magnitude_original = np.abs(X_fft_original)
    magnitude_filtered = np.abs(X_fft_filtered)

    # 计算对应的频率
    N = len(original_signal)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # 绘制频域对比图
    plt.figure(figsize=(12, 6))
    plt.plot(freqs[:N//2], magnitude_original[:N//2], label="Original Signal", color="blue")
    plt.plot(freqs[:N//2], magnitude_filtered[:N//2], label="Filtered Signal", color="red")
    plt.title("Frequency Domain Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

class SpatioFreq():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.0005,
        # learning_rate=0.001,
        learning_rate_sc = 0.01,
        # weight_decay=0.00,
        epochs=1000, 
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10x',
        n_top_genes=1500,
        weight_decay=0.0001,  # L2 正则化系数
        l1_lambda=0.0001,     # L1 正则化系数
        use_frequency_features=True,  # 新增参数：控制是否使用频域特征融合        
        ):
        # 随机种子生成和设置
        if random_seed == 'random':  # 如果种子是'random'，就随机生成一个种子
            self.random_seed = random.randint(1, 10000)
        else:
            self.random_seed = random_seed
        
        print(f"Using random seed: {self.random_seed}")  # 打印使用的随机种子
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        # 这个方法的主要目的是准备模型训练所需的数据，并进行必要的预处理和配置。
        self.use_frequency_features = use_frequency_features  # 保存此参数

        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.l1_lambda=l1_lambda
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.n_top_genes = n_top_genes  # 保存 n_top_genes 参数
        # fix_seed(self.random_seed)

        # if 'highly_variable' not in adata.var.keys():
        #    preprocess(self.adata,n_top_genes=1500)
        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata, n_top_genes=self.n_top_genes)
  
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
        # 提取原始特征1
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        # 如果设置了 use_frequency_features 为 True，则执行频域特征融合
        if use_frequency_features:
            self.integrate_frequency_features()  # 频域特征融合
# 1
    def integrate_frequency_features(self):
        """
        计算频域特征并将其与原始特征矩阵融合，随后降维到与原始特征相同的维度。
        """
        print("Integrating frequency features...")

        # Step 1: 拉普拉斯矩阵
        lap_mtx = get_laplacian_mtx(self.features.cpu().numpy(), num_neighbors=7, normalization=False)

        # Step 2: 频域特征提取
        freq_features, _ = obtain_freq_spots(adata=self.features, lap_mtx=lap_mtx, n_fcs=50, c=1)

        # Step 3: 过滤频率信号
        filtered_freq_features = np.array([
            filter_frequency_signal(freq_signal, method='lowpass', smooth=True, sigma=2) 
            for freq_signal in freq_features.T
        ]).T

        # plot_signals_overall(freq_features, filtered_freq_features, title="Frequency Signal Comparison")
        # Step 4: 特征拼接
        freq_features_tensor = torch.tensor(filtered_freq_features, dtype=torch.float32).to(self.device)
        combined_features = torch.cat((self.features, freq_features_tensor), dim=1)

        # Step 5: PCA降维到原始特征维度
        print("Performing PCA to reduce dimensions...")
        combined_features_np = combined_features.cpu().detach().numpy()
        n_components=self.features.shape[1]
        print(n_components)
        pca = PCA(n_components=self.features.shape[1])  # 降维到原始特征的维度
        # pca = PCA(600)
        reduced_features = pca.fit_transform(combined_features_np)

        # 转回张量并保存
        self.features = torch.tensor(reduced_features, dtype=torch.float32).to(self.device)
        print(f"Features after PCA reduction: {self.features.shape}")


    def train(self):
        # 根据数据类型选择不同的模型
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # L2 正则化
        )

        l1_lambda = 1e-4  # L1 正则化系数

        print('Begin to train ST data...')
        self.model.train()

        # 存储每个 epoch 的损失
        loss_history = []
        # 数据准备：提前计算对比增强特征
        preprocessed_features_a = F.normalize(permutation(self.features), p=2, dim=1).to(self.device)
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            
            # self.features_a = permutation(self.features)
                        # 整理特征
            self.features_a = preprocessed_features_a
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)
            
            # 计算各项损失
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)
            
            # 手动计算 L1 正则化
            l1_norm = sum(param.abs().sum() for param in self.model.parameters())

            # 总损失
            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2) + l1_lambda * l1_norm
            
            # 保存当前 epoch 的损失
            loss_history.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print("Optimization finished for ST data!")

        # # 绘制损失图像
        # plt.figure(figsize=(8, 6))
        # plt.plot(range(self.epochs), loss_history, label='Total Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        with torch.no_grad():
            self.model.eval()
            if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                return self.emb_rec
            else:
                if self.datatype in ['Stereo', 'Slide']:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                    self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                return self.adata

    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        
        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            
            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)
            
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
            
        print("Optimization finished for cell representation learning!")
        
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
         
            return emb_sc
    # 这个方法的主要目的是训练一个映射矩阵，将空间数据和单细胞数据的表示向量映射到同一空间中，以便进行后续的分析或可视化 
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        
        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
          
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
             
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
            
        print("Mapping matrix learning finished!")
        
        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc
    # 这个 loss 方法用于计算模型的损失函数，包括重构损失和对比损失。
    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
# 这个 Noise_Cross_Entropy 方法用于计算噪声对比交叉熵损失。主要目的是衡量预测的空间位置表示与真实的空间位置表示之间的相似度，用于模型的优化和训练过程中。
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    # 这个 cosine_similarity 方法用于计算预测的空间位置表示与真实的空间位置表示之间的余弦相似度矩阵。用于模型评估或结果分析
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M        
