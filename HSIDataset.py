import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import scale


class HSIDataset(Dataset):
    def __init__(self, hsi, gt, patch_size=1):
        '''
        :param hsi: [h, w, bands]
        :param gt: [h, w]
        :param patch_size: scale
        '''
        super(HSIDataset, self).__init__()
        self.hsi = self.add_mirror(hsi, patch_size)  # [h, w, bands]
        self.gt = gt  # [h, w]
        self.patch_size = patch_size
        # 标签数据的索引
        self.indices = tuple(zip(*np.nonzero(gt)))

    # 添加镜像
    @staticmethod
    def add_mirror(data, patch_size):
        dx = patch_size // 2
        if dx != 0:
            h, w, c = data.shape
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, c))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        else:
            mirror = data
        return mirror

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
        :param index:
        :return: 元素光谱信息， 元素的空间信息， 标签
        '''
        x, y = self.indices[index]
        # 领域: [patchsz, patchsz, bands]
        neighbor_region = self.hsi[x: x + self.patch_size, y: y + self.patch_size, :]
        # 类别
        target = self.gt[x, y] - 1
        return torch.tensor(neighbor_region, dtype=torch.float), torch.tensor(target, dtype=torch.long)


# random
# class HSIDatasetPair_(HSIDataset):
#     def __getitem__(self, index):
#         import time
#         begin_time = time.time()
#         x, y = self.indices[index]
#         target = self.gt[x, y]
#         end_time = time.time()
#         print('Finish Time 1: {}'.format(end_time - begin_time))
#         # 正样本
#         begin_time = time.time()
#         pos_indices = tuple(zip(*np.nonzero(self.gt == target)))
#         (x_pos, y_pos) = random.sample(pos_indices, 1)[0]
#         # x_pos, y_pos = pos_indices[random.randint(0, len(pos_indices) - 1)]
#         # 负样本
#         neg_indices = tuple(zip(*np.nonzero(self.gt != target)))
#         (x_neg, y_neg) = random.sample(neg_indices, 1)[0]
#         end_time = time.time()
#         print('Finish Time 2:{}'.format(end_time - begin_time))
#         # x_neg, y_neg = neg_indices[random.randint(0, len(neg_indices) - 1)]
#         # 领域
#         begin_time = time.time()
#         neighbor_region = self.hsi[x: x + self.patch_size, y: y + self.patch_size, :]
#         neighbor_region_pos = self.hsi[x_pos: x_pos + self.patch_size, y_pos: y_pos + self.patch_size, :]
#         neighbor_region_neg = self.hsi[x_neg: x_neg + self.patch_size, y_neg: y_neg + self.patch_size, :]
#         end_time = time.time()
#         print('Finish Time 3: {}'.format(end_time - begin_time))
#         begin_time = time.time()
#         pos_sample = np.stack([neighbor_region, neighbor_region_pos], axis=0)
#         neg_sample = np.stack([neighbor_region, neighbor_region_neg], axis=0)
#
#         sample = np.stack([pos_sample, neg_sample], axis=0)
#         end_time = time.time()
#         print('Finish Time 4: {}'.format(end_time - begin_time))
#         return torch.tensor(sample, dtype=torch.float), torch.tensor([True, False], dtype=torch.long)
#         # return torch.from_numpy(sample), torch.tensor([True, False], dtype=torch.long)


class HSIDatasetPair(HSIDataset):
    def __init__(self, hsi, gt, patch_size=1):
        super().__init__(hsi, gt, patch_size)
        nc = self.gt.max()
        non_zero = gt != 0
        self.pos_pool = [tuple(zip(*np.nonzero(gt == i))) for i in range(1, nc + 1)]
        self.neg_pool = [tuple(zip(*np.nonzero((gt != i) & non_zero))) for i in range(1, nc + 1)]
    def __getitem__(self, index):
        # import time
        # begin_time = time.time()
        x, y = self.indices[index]
        target = self.gt[x, y]
        # end_time = time.time()
        # print('Finish Time 1: {}'.format(end_time - begin_time))
        # # 正样本
        # begin_time = time.time()
        # pos_indices = tuple(zip(*np.nonzero(self.gt == target)))
        # (x_pos, y_pos) = random.sample(pos_indices, 1)[0]
        x_pos, y_pos = random.sample(self.pos_pool[target - 1], 1)[0]
        # x_pos, y_pos = pos_indices[random.randint(0, len(pos_indices) - 1)]
        # 负样本
        # neg_indices = tuple(zip(*np.nonzero(self.gt != target)))
        # (x_neg, y_neg) = random.sample(neg_indices, 1)[0]
        x_neg, y_neg = random.sample(self.neg_pool[target - 1], 1)[0]
        # end_time = time.time()
        # print('Finish Time 2: {}'.format(end_time - begin_time))
        # x_neg, y_neg = neg_indices[random.randint(0, len(neg_indices) - 1)]
        # 领域
        # begin_time = time.time()
        neighbor_region = self.hsi[x: x + self.patch_size, y: y + self.patch_size, :]
        neighbor_region_pos = self.hsi[x_pos: x_pos + self.patch_size, y_pos: y_pos + self.patch_size, :]
        neighbor_region_neg = self.hsi[x_neg: x_neg + self.patch_size, y_neg: y_neg + self.patch_size, :]
        # end_time = time.time()
        # print('Finish Time 3: {}'.format(end_time - begin_time))
        #
        # begin_time = time.time()
        pos_sample = np.stack([neighbor_region, neighbor_region_pos], axis=0)
        neg_sample = np.stack([neighbor_region, neighbor_region_neg], axis=0)

        sample = np.stack([pos_sample, neg_sample], axis=0)
        # end_time = time.time()
        # print('Finish Time 4: {}'.format(end_time - begin_time))
        # print(self.gt[x, y], self.gt[x_pos, y_pos], self.gt[x_neg, y_neg])
        return torch.tensor(sample, dtype=torch.float), torch.tensor([True, False], dtype=torch.long)


class HSIDatasetPairNaive(HSIDataset):
    def __init__(self, hsi, gt, patch_size=1):
        super().__init__(hsi, gt, patch_size)
        # nc = self.gt.max()
        # non_zero = gt != 0
        # self.pos_pool = [tuple(zip(*np.nonzero(gt == i))) for i in range(1, nc + 1)]
        # self.neg_pool = [tuple(zip(*np.nonzero((gt != i) & non_zero))) for i in range(1, nc + 1)]
        self.pair_indices = []
        n = len(self.indices)
        for i in range(n):
            for j in range(i+1, n):
                self.pair_indices.append((i, j))

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, index):
        index1, index2 = self.pair_indices[index]
        x1, y1 = self.indices[index1]
        x2, y2 = self.indices[index2]
        target1, target2 = self.gt[x1, y1], self.gt[x2, y2]
        # x, y = self.indices[index]
        # target = self.gt[x, y]
        # x_pos, y_pos = random.sample(self.pos_pool[target - 1], 1)[0]
        # x_neg, y_neg = random.sample(self.neg_pool[target - 1], 1)[0]

        # 领域
        neighbor_region_1 = self.hsi[x1: x1 + self.patch_size, y1: y1 + self.patch_size, :]
        neighbor_region_2 = self.hsi[x2: x2 + self.patch_size, y2: y2 + self.patch_size, :]
        sample_pair = np.stack([neighbor_region_1, neighbor_region_2], axis=0)

        return torch.tensor(sample_pair, dtype=torch.float), torch.tensor(target1 == target2, dtype=torch.long)


class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt',
        'hidden_size': 17
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt',
            'hidden_size': 12
    },  'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'hidden_size': 14
    },  'Houston':{
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt'
    },  'Indian':{
            'data_key': 'indian_pines_corrected',
            'label_key': 'indian_pines_gt',
            'hidden_size': 12
    },  'Pavia':{
            'data_key': 'pavia',
            'label_key': 'pavia_gt'
    },  'gf5':{
            'data_key': 'gf5',
            'label_key': 'gf5_gt'
        }}


# class Data_prefetcher():
#     def __init__(self, loader: torch.utils.data.DataLoader, device: torch.device):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream(device)
#         # 预加载
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_input, self.next_target = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             self.next_target = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next_input = self.next_input.cuda(non_blocking=True)
#             self.next_target = self.next_target.cuda(non_blocking=True)
#             # # With Amp, it isn't necessary to manually convert data to half.
#             # # if args.fp16:
#             # #     self.next_input = self.next_input.half()
#             # # else:
#             # self.next_input = self.next_input.float()
#             # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
#
#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         input = self.next_input
#         target = self.next_target
#         self.preload()
#         return input, target


# def batch_sampler(dataset, batch_size, shuffle):
#     size = len(dataset)
#     indices = list(range(size))
#     import random
#     if shuffle:
#         random.shuffle(indices)
#     import math
#     batches = int(math.ceil(size/batch_size))
#     for i in range(batches):
#         begin = i * batch_size
#         end = (i + 1) * batch_size
#         data, gt = [], []
#         for j in indices[begin: end]:
#             tmp_data, tmp_gt = dataset[j]
#             data.append(tmp_data)
#             gt.append(tmp_gt)
#         yield torch.stack(data), torch.stack(gt)

# from scipy.io import loadmat
# import numpy as np
# m = loadmat('data/PaviaU/PaviaU.mat')
# data = m['paviaU']
# m = loadmat('data/PaviaU/PaviaU_gt.mat')
# label = m['paviaU_gt']
# data, label = data.astype(np.float32), label.astype(np.long)
# # label += 1
# dataset = HSIDataset(data, label, patch_size=21)
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset, batch_size=128, num_workers=10, shuffle=True)
# import time
# times = []
# iteration = 0
# begin_time = time.time()
# for x, target in loader:
#     end_time = time.time()
#     print('Finish Time {}: {}'.format(iteration, end_time - begin_time))
#     times.append(end_time - begin_time)
#     iteration += 1
#     begin_time = time.time()
# print(np.mean(times))

# times = []
# loader = batch_sampler(dataset, 128, True)
# begin_time = time.time()
# iteration = 0
# for x, target in loader:
#     end_time = time.time()
#     # print(x.shape, target.shape)
#     print('Finish Time {}: {}'.format(iteration, end_time - begin_time))
#     times.append(end_time - begin_time)
#     iteration += 1
#     begin_time = time.time()
# print(np.mean(times))

# for i in range(len(dataset)):
#     begin_time = time.time()
#     x, target = dataset[0]
#     end_time = time.time()
#     print('Finish Time: {}'.format(end_time - begin_time))
# w = data.shape[1]
# index = 150
# l, c = index // w, index % w
# spectra = data[l, c]
# neighbor_region, target = dataset[index]
# print(torch.equal(torch.from_numpy(spectra), neighbor_region[21 // 2, 21 // 2]))
# print(target)
# print(label[l, c])

# from scipy.io import loadmat
# m = loadmat('data/PaviaU/PaviaU.mat')
# data = m['paviaU']
# m = loadmat('data/PaviaU/PaviaU_gt.mat')
# gt = m['paviaU_gt']
# import numpy as np
# data, gt = data.astype(np.float), gt.astype(np.int32)
# dataset = HSIDatasetPair(data, gt, 5)
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
# import time
# times = []
# iteration = 0
# begin_time = time.time()
# for x, target in loader:
#     end_time = time.time()
#     print('Finish Time {}: {}'.format(iteration, end_time - begin_time))
#     times.append(end_time - begin_time)
#     iteration += 1
#     begin_time = time.time()
#     print(x.shape, target.shape)
#     exit(0)
# print(np.mean(times))
# times = []
# iteration = 0
# loader = batch_sampler(dataset, batch_size=128, shuffle=True)
# begin_time = time.time()
# for x, target in loader:
#     end_time = time.time()
#     print('Finish Time {}: {}'.format(iteration, end_time - begin_time))
#     times.append(end_time - begin_time)
#     iteration += 1
#     begin_time = time.time()
# print(np.mean(times))
# import time
# for i in range(len(dataset)):
#     begin_time = time.time()
#     x = dataset[i]
#     end_time = time.time()
#     print('Finish Time: {}'.format(end_time - begin_time))
#     break
# exit(0)
# x, target = dataset[100]
# print('Finish Time: {}'.format(end_time - begin_time))
# print(x.shape, target)
#
# from scipy.io import loadmat
# m = loadmat('data/PaviaU/PaviaU.mat')
# data = m['paviaU']
# m = loadmat('data/PaviaU/PaviaU_gt.mat')
# gt = m['paviaU_gt']
# import numpy as np
# data, gt = data.astype(np.float), gt.astype(np.int32)
# dataset = HSIDatasetPairMultiClass_(data, gt, 7)
# x, target = dataset[0]
# print(x.shape, target)

# data = np.random.rand(9).reshape((3,3,1))
# gt = np.arange(9).reshape((3,3))
# dataset = HSIDatasetPairNaive(data, gt, patch_size=1)
# print(len(dataset))
# print(data)
# x, y = dataset[0]
# print(x, y)