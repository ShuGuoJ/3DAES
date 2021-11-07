import torch
from torch import nn
from utils import pad
from torch.nn import functional as F


# 3层paviaU, patch_size: 7x7
class Encoder(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        '''
        :param in_channel: 表示输入的特征通道数
        :param hidden_size: 表示隐藏层的神经元个数
        :param dim: 表示光谱维的长度
        '''
        super().__init__()
        self.dim = dim
        m = list()
        padding_0 = pad(dim, 5, 3)
        m.append(nn.Conv3d(in_channel, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        padding_2 = pad(dim_2, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        self.out_dim = (dim_2 - 5 + 2 * padding_2) // 3 + 1

        self.encoder = nn.Sequential(*m)

    def forward(self, x):
        assert self.dim == x.shape[2]
        return self.encoder(x)


# 3层 PaviaU, patch_size: 7x7
class Decoder(nn.Module):
    def __init__(self, out_channel, hidden_size, dim):
        super().__init__()
        padding_0 = pad(dim, 5, 3)
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        padding_2 =  pad(dim_2, 5, 3)
        m = list()
        m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        m.append(nn.ConvTranspose3d(hidden_size, out_channel, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(out_channel))
        self.decoder = nn.Sequential(*m)

    def forward(self, x):
        return self.decoder(x)


# 2层paviaU, patch_size: 7x7
class Encoder2(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        '''
        :param in_channel: 表示输入的特征通道数
        :param hidden_size: 表示隐藏层的神经元个数
        :param dim: 表示光谱维的长度
        '''
        super().__init__()
        self.dim = dim
        m = list()
        padding_0 = pad(dim, 5, 3)
        m.append(nn.Conv3d(in_channel, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        # dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        # padding_2 = pad(dim_2, 5, 3)
        # m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0)))
        # m.append(nn.ReLU())
        # m.append(nn.BatchNorm3d(hidden_size))
        self.out_dim = (dim_1 - 5 + 2 * padding_1) // 3 + 1

        self.encoder = nn.Sequential(*m)

    def forward(self, x):
        assert self.dim == x.shape[2]
        return self.encoder(x)


# 2层 PaviaU, patch_size: 7x7
class Decoder2(nn.Module):
    def __init__(self, out_channel, hidden_size, dim):
        super().__init__()
        padding_0 = pad(dim, 5, 3)
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        # dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        # padding_2 =  pad(dim_2, 5, 3)
        m = list()
        # m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0)))
        # m.append(nn.ReLU())
        # m.append(nn.BatchNorm3d(hidden_size))
        m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        m.append(nn.ConvTranspose3d(hidden_size, out_channel, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0)))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(out_channel))
        self.decoder = nn.Sequential(*m)

    def forward(self, x):
        return self.decoder(x)


# 3层 PaviaU, patch_size 7x7 增减对比缓冲层和skip connection
class Net(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder(in_channel, hidden_size, dim)
        n = self.encoder.out_dim * hidden_size
        self.ln = nn.Sequential(
            nn.Linear(n, 2 * n),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(2 * n, n)
        )
        # self.ln = nn.Sequential(
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384)
        # )
        # self.skip_connect = nn.Linear(384, 512)

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        residual = self.ln(h)
        # residual = self.skip_connect(h)
        return h + residual


# leaklyrelu + bottleneck
class Net_Bottleneck3(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder(in_channel, hidden_size, dim)
        n = self.encoder.out_dim * hidden_size
        self.ln = nn.Sequential(
            nn.Linear(n, n // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(n // 2, n // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(n // 2, n)
        )
        # self.ln = nn.Sequential(
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384)
        # )
        # self.skip_connect = nn.Linear(384, 512)

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        residual = self.ln(h)
        # residual = self.skip_connect(h)
        return h + residual


# 多分类
class Net_classification(nn.Module):
    def __init__(self, feature_module, hidden_size, nc):
        super().__init__()
        self.feature_module = feature_module
        vector_length = self.feature_module.encoder.out_dim * hidden_size
        self.classifier = nn.Sequential(nn.Linear(vector_length, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, nc))

    def forward(self, x):
        # x: [batchsz, 1, spectra, height, width]
        h = self.feature_module(x)
        h = h.view((h.shape[0], -1))
        return self.classifier(h)


# 使用bottleneck结构来减少revision模块的参数量
class Net_Bottleneck(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder(in_channel, hidden_size, dim)
        n = self.encoder.out_dim * hidden_size
        self.ln = nn.Sequential(
            nn.Linear(n, n // 2),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(n // 2, n)
        )
        # self.ln = nn.Sequential(

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        residual = self.ln(h)
        # residual = self.skip_connect(h)
        return h + residual



# 深度可分离 encoder
class Net_Depthwise(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder_Depthwise(in_channel, hidden_size, dim)
        n = self.encoder.out_dim * hidden_size
        self.ln = nn.Sequential(
            nn.Linear(n, 2 * n),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(2 * n, n)
        )
        # self.ln = nn.Sequential(
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384)
        # )
        # self.skip_connect = nn.Linear(384, 512)

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        residual = self.ln(h)
        # residual = self.skip_connect(h)
        return h + residual


# skip connection的连接方式该为相乘
class Net_Mul(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder(in_channel, hidden_size, dim)
        n = self.encoder.out_dim * hidden_size
        self.ln = nn.Sequential(
            nn.Linear(n, 2 * n),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(2 * n, n)
        )
        # self.ln = nn.Sequential(
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384),
        #     nn.Sigmoid(),
        #     nn.Linear(384, 384)
        # )
        # self.skip_connect = nn.Linear(384, 512)

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        residual = self.ln(h)
        # residual = self.skip_connect(h)
        return h * residual


# 使用bottleneck结构来减少revision模块的参数量
class Net_Mul_Bottleneck(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        super().__init__()
        self.encoder = Encoder(in_channel, hidden_size, dim)
        n = self.encoder.out_dim * hidden_size
        self.ln = nn.Sequential(
            nn.Linear(n, n // 2),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(n // 2, n)
        )

    def forward(self, x):
        batch = x.shape[0]
        h = self.encoder(x)
        h = F.adaptive_max_pool3d(h, (None, 1, 1))
        h = h.view((batch, -1))
        residual = self.ln(h)
        # residual = self.skip_connect(h)
        return h * residual

# 3层paviaU, patch_size: 7x7
class Encoder_Depthwise(nn.Module):
    def __init__(self, in_channel, hidden_size, dim):
        '''
        :param in_channel: 表示输入的特征通道数
        :param hidden_size: 表示隐藏层的神经元个数
        :param dim: 表示光谱维的长度
        '''
        super().__init__()
        self.dim = dim
        m = list()
        padding_0 = pad(dim, 5, 3)
        m.append(nn.Conv3d(in_channel, in_channel, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0), groups=in_channel))
        m.append(nn.Conv3d(in_channel, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        padding_2 = pad(dim_2, 5, 3)
        m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        self.out_dim = (dim_2 - 5 + 2 * padding_2) // 3 + 1

        self.encoder = nn.Sequential(*m)

    def forward(self, x):
        assert self.dim == x.shape[2]
        return self.encoder(x)


# 3层 PaviaU, patch_size: 7x7
class Decoder_Depthwise(nn.Module):
    def __init__(self, out_channel, hidden_size, dim):
        super().__init__()
        padding_0 = pad(dim, 5, 3)
        dim_1 = (dim - 5 + 2 * padding_0) // 3 + 1
        padding_1 = pad(dim_1, 5, 3)
        dim_2 = (dim_1 - 5 + 2 * padding_1) // 3 + 1
        padding_2 =  pad(dim_2, 5, 3)
        m = list()
        m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_2, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_1, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(hidden_size))
        m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(padding_0, 0, 0), groups=hidden_size))
        m.append(nn.Conv3d(hidden_size, out_channel, 1, 1))
        m.append(nn.ReLU())
        m.append(nn.BatchNorm3d(out_channel))
        self.decoder = nn.Sequential(*m)

    def forward(self, x):
        return self.decoder(x)

# # 3层paviaU, patch_size: 7x7 深度可分离
# class Encoder_Depthwise(nn.Module):
#     def __init__(self, in_channel, hidden_size):
#         super().__init__()
#         m = list()
#         m.append(nn.Conv3d(in_channel, in_channel, (5, 3, 3), (3, 1, 1), groups=in_channel))
#         m.append(nn.Conv3d(in_channel, hidden_size, 1, 1))
#         m.append(nn.ReLU())
#         m.append(nn.BatchNorm3d(hidden_size))
#         m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(2, 0, 0), groups=hidden_size))
#         m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
#         m.append(nn.ReLU())
#         m.append(nn.BatchNorm3d(hidden_size))
#         m.append(nn.Conv3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(1, 0, 0), groups=hidden_size))
#         m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
#         m.append(nn.ReLU())
#         m.append(nn.BatchNorm3d(hidden_size))
#
#         self.encoder = nn.Sequential(*m)
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# # 3层 PaviaU, patch_size: 7x7 深度可分离
# class Decoder_Depthwise(nn.Module):
#     def __init__(self, out_channel, hidden_size):
#         super().__init__()
#         m = list()
#         m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(1, 0, 0),
#                                     output_padding=(2, 0, 0), groups=hidden_size))
#         m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
#         m.append(nn.ReLU())
#         m.append(nn.BatchNorm3d(hidden_size))
#         m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), padding=(2, 0, 0),
#                                     output_padding=(2, 0, 0), groups=hidden_size))
#         m.append(nn.Conv3d(hidden_size, hidden_size, 1, 1))
#         m.append(nn.ReLU())
#         m.append(nn.BatchNorm3d(hidden_size))
#         m.append(nn.ConvTranspose3d(hidden_size, hidden_size, (5, 3, 3), (3, 1, 1), output_padding=(2, 0, 0)))
#         m.append(nn.Conv3d(hidden_size, out_channel, 1, 1))
#         m.append(nn.ReLU())
#         m.append(nn.BatchNorm3d(out_channel))
#         self.decoder = nn.Sequential(*m)
#
#     def forward(self, x):
#         return self.decoder(x)

# bands = 103
# net = Encoder2(1, 64, bands)
# print(net)
# input = torch.rand(1, 1, bands, 7, 7)
# out = net(input)
# print(out.shape)

# net = Encoder(1, 64, 224)
# print(net)
# input = torch.rand(1, 1, 224, 7, 7)
# out = net(input)
# print(out.shape)

# net = Decoder2(1, 64, 103)
# print(net)
# input = torch.rand(1, 64, 11, 3, 3)
# out = net(input)
# print(out.shape)

# net = Decoder(1, 64, 224)
# print(net)
# input = torch.rand(1, 64, 8, 1, 1)
# out = net(input)
# print(out.shape)

# net = Encoder_Depthwise(1, 128, 204)
# print(net)
# input = torch.rand(1, 1, 204, 7, 7)
# out = net(input)
# print(out.shape)

# net = Decoder_Depthwise(1, 128, 103)
# print(net)
# input = torch.rand(1, 128, 3, 1, 1)
# out = net(input)
# print(out.shape)

# net = Net_mul(1, 128, 103)
# # net = Net(1, 128, 103)
# net.eval()
# input = torch.rand((1, 1, 103, 7, 7))
# out = net(input)
# print(out.shape)

# net = Net_Depthwise(1, 128, 103)
# net.eval()
# input = torch.rand((1,1,103,7,7))
# out = net(input)
# print(out.shape)