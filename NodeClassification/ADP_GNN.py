from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

class ADP_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(ADP_prop, self).__init__(aggr='add', **kwargs)  # 使用加法聚合
        self.K = K
        # 定义一个可训练的混合参数，初始值设为0.5，后续通过 sigmoid 保证其在 (0,1) 范围内
        self.alpha = Parameter(torch.tensor(0.5))
        # 多项式组合的系数，目前同一组系数用于两段式多项式（Taylor 与 Bernstein 部分）
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # 将多项式参数全部初始化为1（你也可以尝试其它初始化方式）
        self.temp.data.fill_(1)
        # 初始化混合参数
        self.alpha.data.fill_(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        # 保存原始输入，用于第二部分的计算
        x0 = x

        # 对多项式系数使用 ReLU，确保非负
        TEMP = F.relu(self.temp)
        # 对 alpha 使用 sigmoid 激活，使其值位于 (0,1) 之间
        alpha = torch.sigmoid(self.alpha)

        # 计算归一化后的拉普拉斯算子 L = I - D^(-0.5) A D^(-0.5)
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization='sym', dtype=x.dtype,
            num_nodes=x.size(self.node_dim)
        )

        # -------------------------------
        # 第一部分多项式（Taylor-like）输出计算，与原来一致
        tmp = [x]
        x_tmp = x
        for i in range(self.K):
            x_tmp = self.propagate(edge_index1, x=x_tmp, norm=norm1, size=None)
            tmp.append(x_tmp)

        out = (1 / math.factorial(self.K)) * TEMP[self.K] * tmp[self.K]
        for i in range(self.K):
            out += (1 / math.factorial(self.K - i - 1)) * TEMP[self.K - i - 1] * tmp[self.K - i - 1]

        # -------------------------------
        # 修改后的第二部分多项式（Bernstein-like）输出
        # 新表达式为：∑_{k=0}^{K} (comb(K, k)/2^K) * TEMP[k] * (2I-L)^(K-k) L^k x

        # 第一步：计算 L^k x 的序列 (k=0,...,K)
        L_series = [x0]
        x_l = x0
        for k in range(1, self.K + 1):
            x_l = self.propagate(edge_index1, x=x_l, norm=norm1, size=None)
            L_series.append(x_l)

        # 第二步：计算(2I-L)的边信息
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2., num_nodes=x0.size(self.node_dim)
        )

        second_part = 0
        # 对于每个 k，先取 L^k x，再应用 (2I-L)^(K-k)
        for k in range(self.K + 1):
            y = L_series[k]
            for _ in range(self.K - k):
                y = self.propagate(edge_index2, x=y, norm=norm2, size=None)
            second_part = second_part + (comb(self.K, k) / (2 ** self.K)) * TEMP[k] * y

        # 使用 learnable 混合参数 alpha 将两部分进行线性组合
        out = alpha * out + (1 - alpha) * second_part

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={}, alpha={})'.format(
            self.__class__.__name__, self.K,
            self.temp, torch.sigmoid(self.alpha))




# class ADP_prop(MessagePassing):
#     def __init__(self, K, bias=True, alpha = 0.9, **kwargs):
#         super(ADP_prop, self).__init__(aggr='add', **kwargs)#aggr所用的聚合方法
#         # 当传入字典形式的参数时，就要使用 ** kwargs
#         self.K = K
#         self.alpha = alpha
#         # 定义新的初始化变量。模型中的参数，它是Parameter()类，
#         # 先转化为张量，再转化为可训练的Parameter对象
#         # Parameter用于将参数自动加入到参数列表
#         self.temp = Parameter(torch.Tensor(self.K + 1))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.temp.data.fill_(1)#Fills self tensor with the specified value.
#
#     def forward(self, x,  edge_index, edge_weight=None):
#
#         #TEMP = self.temp
#         TEMP = F.relu(self.temp)
#         alpha = self.alpha
#
#         # 计算拉普拉斯算子L=I-D^(-0.5)AD^(-0.5)
#         edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
#                                            num_nodes=x.size(self.node_dim))
#
#         # 初始化 tmp 列表
#         tmp = [x]
#         for i in range(self.K):
#             x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
#             tmp.append(x)
#
#         # 第一段多项式的输出
#         out = (1 / math.factorial(self.K)) * TEMP[self.K] * tmp[self.K]
#
#         # 第二段多项式的输出
#         for i in range(self.K):
#             x = tmp[self.K - i - 1]
#             out += (1 / math.factorial(self.K - i - 1)) * TEMP[self.K - i - 1] * x
#
#         # 计算第二段的输出，需重新计算 edge_index2 和 norm2 2I-L
#         edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))
#
#         # 清空 tmp 列表以重新计算
#         tmp = [x]
#         for i in range(self.K):
#             x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
#             tmp.append(x)
#
#         # 第二段多项式的初始项
#         out = alpha*out + (1-alpha)*(comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]
#
#         # 合并第二段的项
#         for i in range(self.K):
#             x = tmp[self.K - i - 1]
#             x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
#             for j in range(i):
#                 x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
#
#             out = out + (1-alpha)*(comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
#
#         return out
#
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
#
#     def __repr__(self):
#         return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
#                                           self.temp)
#
#
#
#

