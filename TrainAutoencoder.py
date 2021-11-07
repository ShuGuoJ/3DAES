"""训练autoencoder模型"""
import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils import weight_init, train_test_split
from HSIDataset import HSIDataset, DatasetInfo
from Model.module import Encoder, Decoder, Encoder_Depthwise, Decoder_Depthwise, Encoder2, Decoder2
from torch.utils.data import DataLoader, random_split
from Trainer import Trainer
import os
import argparse
from visdom import Visdom
from sklearn.preprocessing import scale
from Monitor import GradMonitor
from visualize import visualize, reduce_dimension_

isExists = lambda path: os.path.exists(path)
EPOCHS = 10
LR = 1e-1
BATCHSZ = 128
NUM_WORKERS = 10
SEED = 666
torch.manual_seed(SEED)
# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom(port=17000)
# SAVE_PATH = 'models/paviaU/encoder'
PATCH_SIZE = 19
HIDDEN_SIZE = 128

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='Dataset name')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Training epoch')
    parser.add_argument('--gpu', type=int, nargs='*', default=[2,3],
                        help='gpu id')
    arg = parser.parse_args()
    dataset_name = arg.name
    gpu_id = arg.gpu
    if isinstance(gpu_id, list) and len(gpu_id) > 1:
        device = [torch.device('cuda:{}'.format(id)) for id in gpu_id]
    elif isinstance(gpu_id, list) and len(gpu_id) == 1:
        device = torch.device('cuda:{}'.format(gpu_id[0]))
    else:
        device = torch.device('cuda:{}'.format(gpu_id))
    # 保存路径
    save_path = 'models/{}/encoder3_{}_{}'.format(dataset_name, HIDDEN_SIZE, PATCH_SIZE)
    # 绘画loss, mse_loss和constrative_loss图
    viz.line([[0., 0.]], [0], win='{}_encoder_loss'.format(dataset_name), opts={'title': '{}_loss'.format(dataset_name),
                                                'legend': ['train', 'test']})
    viz.line([0.], [0], win='{}_encoder_grad'.format(dataset_name), opts={'title': '{}_grad'.format(dataset_name)})
    # 加载数据集
    info = DatasetInfo.info[dataset_name]
    m = loadmat('data/{0}/{0}.mat'.format(dataset_name))
    data = m[info['data_key']]
    m = loadmat('data/{0}/{0}_gt.mat'.format(dataset_name))
    gt = m[info['label_key']]
    data, gt = data.astype(np.float), gt.astype(np.int32)
    # 数据标准化
    h, w, c = data.shape
    data = data.reshape((h*w, c))
    data = scale(data)
    data = data.reshape((h, w, c))
    # 划分样本 80%训练 20%测试
    # 构造数据集
    dataset = HSIDataset(data, gt+1, patch_size=PATCH_SIZE)
    train_size = int(len(dataset) * 0.8)
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCHSZ, num_workers=NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSZ, num_workers=NUM_WORKERS)
    # 构造模型，优化器，损失函数
    # hidden size: 128
    # 3层
    encoder = Encoder(1, HIDDEN_SIZE, c)
    decoder = Decoder(1, HIDDEN_SIZE, c)
    # 2层
    # encoder = Encoder2(1, HIDDEN_SIZE, c)
    # decoder = Decoder2(1, HIDDEN_SIZE, c)
    # encoder = Encoder_Depthwise(1, 128)
    # decoder = Decoder_Depthwise(1, 128)
    net = nn.Sequential(encoder, decoder)
    if isinstance(device, list):
        net = nn.DataParallel(net, device)
        device = device[0]
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    # 构造训练器
    trainer = Trainer(net)
    # 构造梯度监控器
    monitor = GradMonitor()
    # min_loss = 1e10
    # 训练模型
    print('*'*5 + dataset_name + '*'*5)
    print('*'*5 + str(HIDDEN_SIZE) + '*'*5)
    print('*'*5 + str(PATCH_SIZE) + '*'*5)
    for epoch in range(arg.epoch):
        print('***** EPOCH: {} *****'.format(epoch))
        # import time
        # begin_time = time.time()
        train_loss, grad = trainer.train(train_loader, optimizer, criterion,
                                         device, monitor)
        test_loss = trainer.evaluate(test_loader, criterion, device)
        # end_time = time.time()
        # print('Finish Time: {}'.format(end_time - begin_time))
        # exit(0)
        print('train loss: {}'.format(train_loss))
        print('test loss: {}'.format(test_loss))
        # 绘画曲线图
        viz.line([[train_loss, test_loss]], [epoch], win='{}_encoder_loss'.format(dataset_name), update='append')
        viz.line([grad], [epoch], win='{}_encoder_grad'.format(dataset_name), update='append')
        # if test_loss < min_loss:
        #     if not isExists(save_path):
        #         os.makedirs(save_path)
        #     torch.save(encoder, os.path.join(save_path, 'best.pkl'))
        #     min_loss = test_loss
        if not isExists(save_path):
            os.makedirs(save_path)
        torch.save(encoder.state_dict(), os.path.join(save_path, '{}.pkl'.format(epoch)))

    print('***** FINISH *****')
    # 绘画样本点分布
    # net = torch.load(os.path.join(SAVE_PATH, 'best.pkl'))
    # net.eval()
    # net.to(DEVICE)
    # n_h, n_gt = [], []
    # gt_ = train_test_split(gt, 900)
    # dataset = HSIDataset(data, gt, patch_size=PATCH_SIZE)
    # loader = DataLoader(dataset, batch_size=128, num_workers=64)
    # for x, target in loader:
    #     x = x.to(DEVICE)
    #     x = x.permute(0, 3, 1, 2).unsqueeze(1)
    #     with torch.no_grad():
    #         features = net(x)
    #     b = features.shape[0]
    #     features = features.reshape((b, -1))
    #     n_h.append(features)
    #     n_gt.append(target)
    # n_h = torch.cat(n_h)
    # n_h = reduce_dimension_(n_h)
    # n_gt = torch.cat(n_gt, dim=0)
    # visualize(n_h, n_gt)





