import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GATConv
''' 定义了一个判别器类 Discriminator,用于区分正样本和负样本。
这个类的作用是接收两个隐藏层表示和一个聚合的图级别表示作为输入，计算正样本和负样本之间的得分，
并将结果返回给模型进行进一步处理，如对比损失的计算。'''
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    # 定义了一个平均池化读出层 AvgReadout 类，用于从节点表示中计算图级别的表示。

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    # 实现了一个基本的图卷积网络（GCN）模型，用于学习节点的表示
class Encoder1(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        # 初始化权重矩阵的参数
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
# 定义模型的前向传播过程
    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a
class Encoder2(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        # 权重参数
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        # 残差连接时，添加一个匹配维度的线性变换
        if self.in_features != self.out_features:
            self.residual_fc = nn.Linear(self.in_features, self.out_features)
        else:
            self.residual_fc = None
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # 输入的残差分支
        res_feat = feat
        if self.residual_fc is not None:
            res_feat = self.residual_fc(res_feat)  # 如果维度不匹配，则进行线性变换
        
        # 正向传播主分支
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        # 添加残差连接
        z = z + res_feat  # 残差连接，确保维度一致

        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)  # 残差后的激活输出
        
        # 第二条路径用于辅助特征 feat_a
        res_feat_a = feat_a
        if self.residual_fc is not None:
            res_feat_a = self.residual_fc(res_feat_a)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        
        # 添加残差连接
        z_a = z_a + res_feat_a

        emb_a = self.act(z_a)
        
        # 图的全局特征
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        # 对抗损失计算
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a


# class SelfAttention(nn.Module):
#     def __init__(self, hidden_dims, dropout=0.1):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(hidden_dims, hidden_dims)
#         self.key = nn.Linear(hidden_dims, hidden_dims)
#         self.value = nn.Linear(hidden_dims, hidden_dims)
#         self.scale = hidden_dims ** -0.5
#         self.dropout = nn.Dropout(p=dropout)  # Dropout regularization

#         # Weight initialization
#         torch.nn.init.xavier_uniform_(self.query.weight)
#         torch.nn.init.xavier_uniform_(self.key.weight)
#         torch.nn.init.xavier_uniform_(self.value.weight)

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)  

#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)

#         attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
#         attn_weights = self.dropout(attn_weights)  # Apply dropout
#         out = torch.bmm(attn_weights, V)

#         if out.size(1) == 1:
#             out = out.squeeze(1)
#         return out
class SelfAttention(nn.Module):
    def __init__(self, hidden_dims, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dims, hidden_dims)
        self.key = nn.Linear(hidden_dims, hidden_dims)
        self.value = nn.Linear(hidden_dims, hidden_dims)
        self.scale = hidden_dims ** -0.5
        self.dropout = nn.Dropout(p=dropout)  # Dropout regularization

        # Weight initialization
        torch.nn.init.xavier_uniform_(self.query.weight)
        torch.nn.init.xavier_uniform_(self.key.weight)
        torch.nn.init.xavier_uniform_(self.value.weight)

        # 如果输入和输出维度不同，则用线性变换调整
        self.residual_fc = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        attn_weights = self.dropout(attn_weights)  # Apply dropout
        out = torch.bmm(attn_weights, V)

        # 残差连接：输入和注意力输出相加
        if out.size(-1) != x.size(-1):  # 如果维度不匹配，调整维度
            x = self.residual_fc(x)
        out = out + x  # 残差连接

        if out.size(1) == 1:
            out = out.squeeze(1)
        return out




class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, use_attention=True):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.use_attention = use_attention  # 是否使用注意力机制

        # 权重参数
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        # 残差连接时，添加一个匹配维度的线性变换
        if self.in_features != self.out_features:
            self.residual_fc = nn.Linear(self.in_features, self.out_features)
        else:
            self.residual_fc = None

        # 注意力模块
        if self.use_attention:
            self.attention = SelfAttention(hidden_dims=in_features)

        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # 输入的残差分支
        res_feat = feat
        if self.residual_fc is not None:
            res_feat = self.residual_fc(res_feat)  # 如果维度不匹配，则进行线性变换

        # 注意力机制融合：对输入特征进行加权
        if self.use_attention:
            feat = self.attention(feat)  # 使用注意力机制处理输入特征

        # 正向传播主分支
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)

        # 添加残差连接
        z = z + res_feat  # 残差连接，确保维度一致
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        emb = self.act(z)  # 残差后的激活输出

        # 第二条路径用于辅助特征 feat_a
        res_feat_a = feat_a
        if self.residual_fc is not None:
            res_feat_a = self.residual_fc(res_feat_a)

        # 注意力机制融合：对辅助特征进行加权
        if self.use_attention:
            feat_a = self.attention(feat_a)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)

        # 添加残差连接
        z_a = z_a + res_feat_a
        emb_a = self.act(z_a)

        # 图的全局特征
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # 对抗损失计算
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a


class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a =self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a     
# 定义了一个名为 Encoder_sc 的神经网络模型类，用于对单细胞数据进行编码和解码。
# 这个类实现了一个简单的自编码器模型，用于对单细胞数据进行特征提取和重构，
class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        #self.linear1 = torch.nn.Linear(self.dim_input, self.dim_output)
        #self.linear2 = torch.nn.Linear(self.dim_output, self.dim_input)
        
        #self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim_output))
        #self.weight1_de = Parameter(torch.FloatTensor(self.dim_output, self.dim_input))
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        #x = self.linear1(x)
        #x = self.linear2(x)
        
        #x = torch.mm(x, self.weight1_en)
        #x = torch.mm(x, self.weight1_de)
        
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x
    # 这段代码定义了一个名为 Encoder_map 的神经网络模型类，用于学习一个从单细胞到空间位置的映射矩阵。
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 
