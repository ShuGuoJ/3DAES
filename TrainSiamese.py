"""训练孪生网络, 并每次保存模型的参数"""
import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils import weight_init, train_test_split
from HSIDataset import HSIDatasetPair, DatasetInfo, HSIDataset
from Model.module import Encoder, Net, Net_Mul, Net_Depthwise
from torch.utils.data import DataLoader
from Trainer import PairTrainer
import os
import argparse
from visdom import Visdom
from sklearn.preprocessing import scale
from Monitor import GradMonitor
from torch.utils.data import random_split
from visualize import visualize, reduce_dimension_

isExists = lambda path: os.path.exists(path)
EPOCHS = 10
LR = 1e-2
BATCHSZ = 10
'''
对于读取高光谱数据集，增加进程数会降低的读取速度。这有可能是因为与图像数据集不同，在创建完
dataset之后，数据就一直被存储在内存之中。所以增加进程数读取数据可能会导致大多数的时间花费在
创建进程的开销上。经过实验，若读取batch_size=10的高光谱数据集，num_workers=0会获得8加速比，
相比于num_workers=64
'''
# NUM_WORKERS = 64
SEED = 666
torch.manual_seed(SEED)
# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom(port=17000)
torch.multiprocessing.set_sharing_strategy('file_system')
# SAVE_PATH = 'models/paviaU/encoder3层+contrastive+10+2l+random+skip+sigmoid+depthwise'
PATCH_SIZE = 13
HIDDEN_SIZE = 64
RUN = 10
SAMPLES = 9


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train contrastive model')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Training epoch')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='Dataset name')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu id')
    arg = parser.parse_args()
    dataset_name = arg.name
    gpu_id = arg.gpu
    # 预训练模型路径
    pretraining_path = 'models/{}/encoder3_{}_{}/best.pkl'.format(dataset_name, HIDDEN_SIZE, PATCH_SIZE)
    # 保存根路径
    save_root_path = 'models/{}/siamese3_{}_{}_{}'.format(dataset_name, HIDDEN_SIZE, PATCH_SIZE, SAMPLES)
    device = torch.device('cuda:{}'.format(gpu_id))
    # 加载数据集
    info = DatasetInfo.info[dataset_name]
    m = loadmat('data/{0}/{0}.mat'.format(dataset_name))
    data = m[info['data_key']]
    data = data.astype(np.float)
    # 数据标准化
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = scale(data)
    data = data.reshape((h, w, c))
    # 加载预训练模型
    # pretrained_encoder = torch.load(pretraining_path)
    encoder = Net(1, HIDDEN_SIZE, c)
    encoder.encoder.load_state_dict(torch.load(pretraining_path, map_location='cpu'))
    # 固定encoder参数
    for p in encoder.encoder.parameters():
        p.requires_grad = False
    pair_vector_length = 2 * encoder.encoder.out_dim * HIDDEN_SIZE
    metric_module = nn.Sequential(nn.Linear(pair_vector_length, pair_vector_length // 2),
                                  nn.ReLU(),
                                  nn.Linear(pair_vector_length // 2, pair_vector_length // 2),
                                  nn.ReLU(),
                                  nn.Linear(pair_vector_length // 2, 2))
    net = nn.Sequential(encoder, metric_module)
    print('*'*5 + 'SAMPLE: {}'.format(SAMPLES) + '*'*5)
    # 训练训练10组数据
    for r in range(RUN):
        # 重置模型参数
        encoder.apply(weight_init)
        metric_module.apply(weight_init)
        # 保存路径
        save_path = os.path.join(save_root_path, str(r))
        # 绘画loss, mse_loss和constrative_loss图
        viz.line([[0., 0., 0.]], [0], win='{} loss&acc {}'.format(dataset_name, r),
                 opts={'title': '{} loss&acc {}'.format(dataset_name, r),
                       'legend': ['train', 'test', 'accuracy']})
        viz.line([0.], [0], win='{} grad {}'.format(dataset_name, r),
                 opts={'title': '{} grad {}'.format(dataset_name, r)})
        # viz.line([0.], [0], win='{} accuracy {}'.format(dataset_name, r),
        #          opts={'title': '{} accuracy {}'.format(dataset_name, r)})
        print('*'*5 + 'RUN {}'.format(r) + '*'*5)
        # 读取训练样本和测试样本的标签
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(dataset_name, SAMPLES, r))
        train_gt, test_gt = m['train_gt'], m['test_gt']
        # data, gt = data.astype(np.float), gt.astype(np.int32)
        train_gt, test_gt = train_gt.astype(np.int32), test_gt.astype(np.int32)

        import random
        random.seed(971104)
        te_index = tuple(zip(*np.nonzero(test_gt)))
        te_index = random.sample(te_index, 100)
        tmp = np.zeros_like(test_gt)
        te_index = tuple(zip(*te_index))
        tmp[te_index] = test_gt[te_index]
        test_gt = tmp
        # 构造数据集
        train_dataset = HSIDatasetPair(data, train_gt, patch_size=PATCH_SIZE)
        test_dataset = HSIDatasetPair(data, test_gt, patch_size=PATCH_SIZE)
        if dataset_name == 'gf5':
            length = len(test_dataset)
            size = int(0.6 * length)
            test_dataset, _ = random_split(test_dataset, [size, length - size])
        train_loader = DataLoader(train_dataset, batch_size=10, num_workers=2, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCHSZ, num_workers=4, pin_memory=True)
        # dataset = HSIDataset(data, test_gt, patch_size=PATCH_SIZE)
        # loader = DataLoader(dataset, batch_size=BATCHSZ)
        # 绘画原始样本分布
        # indices = np.nonzero(test_gt)
        # spectrum = data[indices]
        # spectrum = reduce_dimension_(spectrum)
        # viz.scatter(torch.from_numpy(spectrum), torch.from_numpy(test_gt[indices]),
        #             win='raw', opts={'title': 'raw data',
        #                              'legend': [str(i) for i in range(gt.max())]})
        # 构造二分类器，优化器，损失函数
        optimizer = optim.Adam(net.parameters())
        # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        # 构造训练器
        # trainer = PairTrainer(net)
        trainer = PairTrainer(net)
        # 构造梯度监控器
        monitor = GradMonitor()
        # 训练模型
        losses = np.zeros((2, arg.epoch), dtype=np.float)
        for epoch in range(arg.epoch):
            print('***** EPOCH: {} *****'.format(epoch))
            train_loss, grad = trainer.train(train_loader, optimizer, criterion,
                                             device, monitor)
            test_loss, acc = trainer.evaluate(test_loader, criterion, device)
            print('train loss: {} test_loss: {} acc: {}'.format(train_loss, test_loss, acc))
            # 绘画曲线图
            viz.line([[train_loss, test_loss, acc]], [epoch],
                     win='{} loss&acc {}'.format(dataset_name, r), update='append')
            viz.line([grad], [epoch], win='{} grad {}'.format(dataset_name, r), update='append')
            # viz.line([acc], [epoch], win='{} accuracy'.format(dataset_name), update='append')

            # if not isExists(save_path):
            #     os.makedirs(save_path)
            # torch.save(encoder.state_dict(), os.path.join(save_path, '{}.pkl'.format(epoch)))
            losses[0, epoch], losses[1, epoch] = train_loss, test_loss
        np.save('{}_random.npy'.format(arg.name), losses)
        exit(0)
            # if (epoch + 1) % 2 == 0:
            #     torch.save(encoder.state_dict(), os.path.join(SAVE_PATH, '{}.pkl'.format(epoch)))
            # if (epoch + 1) % 5 == 0:
            #     # 绘画样本点分布
            #     encoder.eval()
            #     encoder.to(DEVICE)
            #     n_h, n_gt = [], []
            #     for x, target in loader:
            #         x = x.to(DEVICE)
            #         x = x.permute(0, 3, 1, 2).unsqueeze(1)
            #         with torch.no_grad():
            #             features = encoder(x)
            #         b = features.shape[0]
            #         features = features.reshape((b, -1))
            #         n_h.append(features)
            #         n_gt.append(target)
            #     n_h = torch.cat(n_h, dim=0)
            #     n_h = reduce_dimension_(n_h)
            #     n_gt = torch.cat(n_gt, dim=0)
            #     viz.scatter(n_h, n_gt + 1, win='distribution_', opts={'title': 'Distribution_epoch_{}'.format(epoch),
            #                                                           'legend': [str(i) for i in range(gt.max())]
            #                                                           })

        print('***** FINISH *****')
    # 绘画样本点分布
    # net = torch.load(os.path.join(SAVE_PATH, 'best.pkl'))
    # net.eval()
    # net.to(DEVICE)
    # n_h, n_gt = [], []
    # gt_ = train_test_split(gt, 900)
    # dataset = HSIDataset(data, gt, patch_size=PATCH_SIZE)
    # loader = DataLoader(dataset, batch_size=128, num_workers=32)
    # for i, (x, target) in enumerate(loader):
    #     x = x.to(DEVICE)
    #     # sample_1, sample_2 = torch.split(x, 1, dim=1)
    #     # sample_1, sample_2 = sample_1.squeeze(1), sample_2.squeeze(1)
    #     # x = torch.cat([sample_1, sample_2], dim=0)
    #     x = x.permute(0, 3, 1, 2).unsqueeze(1)
    #     with torch.no_grad():
    #         features = net(x)
    #     b = features.shape[0]
    #     features = features.reshape((b, -1))
    #     n_h.append(features)
    #     n_gt.append(target)
    # #     if i%500==0:
    # #         print('Finish')
    # # print('Finish')
    # # exit(0)
    # n_h = torch.cat(n_h)
    # n_h = reduce_dimension_(n_h)
    # n_gt = torch.cat(n_gt, dim=0)
    # visualize(n_h, n_gt)





