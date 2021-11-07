import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import optimizer as optimizer_
from Loss import ContrastiveLoss
from torch.nn import utils
# from HSIDataset import Data_prefetcher


class Trainer(object):
    r"""模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 训练过程
    def train(self, data_loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device, monitor=None):
        self.model.train()
        if isinstance(self.model, torch.nn.DataParallel):
            encoder = torch.nn.DataParallel(self.model.module[0], self.model.device_ids)
            decoder = torch.nn.DataParallel(self.model.module[1], self.model.device_ids)
            encoder.to(device)
            decoder.to(device)
        else:
            self.model.to(device)
            encoder = self.model[0]
            decoder = self.model[1]
        criterion.to(device)
        losses = []
        grads = []
        for step, (x, target) in enumerate(data_loader):
            x, target = x.to(device), target.to(device)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            # 高斯噪声
            e = torch.randn_like(x)
            x_ = x + e
            h = encoder(x_)
            x_hat = decoder(h)
            # 计算mse_loss
            loss = criterion(x_hat, x)  # 计算损失函数值
            # 反向传播
            optimizer.zero_grad()
            # 梯度裁剪
            utils.clip_grad_value_(self.get_parameters(), 0.00001)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # 监控器
            if monitor is not None:
                grads.append(monitor(self.get_parameters()))
            if (step+1) % 5 == 0:
                print('batch:{} loss:{:.6f}'.format(step, loss.item()))
        if len(grads) > 0:
            grads_mean = np.mean(grads)
        else:
            grads_mean = None
        return np.mean(losses), grads_mean

    # 验证过程
    def evaluate(self, data_loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        if isinstance(self.model, torch.nn.DataParallel):
            encoder = torch.nn.DataParallel(self.model.module[0], self.model.device_ids)
            decoder = torch.nn.DataParallel(self.model.module[1], self.model.device_ids)
            # encoder.cuda()
            # decoder.cuda()
        else:
            self.model.to(device)
            encoder = self.model[0]
            decoder = self.model[1]
        criterion.to(device)
        losses = []
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            with torch.no_grad():
                h = encoder(x)
                x_hat = decoder(h)
            # 计算mse_loss
            loss = criterion(x_hat, x)  # 计算损失函数值
            # 计算
            losses.append(loss.item())
        return np.mean(losses)

    def get_parameters(self):
        return self.model.parameters()


# random
class PairTrainer(object):
    r"""
    Matching模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 训练过程
    def train(self, data_loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device, monitor=None):
        self.model.train()
        self.model.to(device)
        encoder = self.model[0]
        encoder.encoder.eval()
        metric_module = self.model[1]
        criterion.to(device)
        losses = []
        grads = []
        for step, (x, target) in enumerate(data_loader):
            x, target = x.to(device), target.to(device)
            x = torch.cat([sample.squeeze() for sample in torch.split(x, 1)])
            target = target.flatten()
            # 拆分样本
            batch = x.shape[0]
            sample_1, sample_2 = torch.split(x, 1, dim=1)
            sample_1, sample_2 = sample_1.squeeze(1), sample_2.squeeze(1)
            x = torch.cat([sample_1, sample_2], dim=0)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            h = encoder(x)
            h = h.reshape((h.shape[0], -1))
            sample_1, sample_2 = torch.split(h, batch, dim=0)
            pairs = torch.cat([sample_1, sample_2], dim=-1)
            similarity = metric_module(pairs)
            # 计算crossentropy
            loss = criterion(similarity, target)  # 计算损失函数值
            # 反向传播
            optimizer.zero_grad()
            # # 梯度裁剪
            # utils.clip_grad_value_(self.get_parameters(), 0.00001)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # 监控器
            if monitor is not None:
                grads.append(monitor(self.get_parameters()))
            if (step+1) % 5 == 0:
                print('batch:{} loss:{:.6f}'.format(step, loss.item()))
            losses.append(loss.item())
        if len(grads) > 0:
            grads_mean = np.mean(grads)
        else:
            grads_mean = None
        return np.mean(losses), grads_mean

    # 验证过程
    def evaluate(self, data_loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        self.model.to(device)
        criterion.to(device)
        losses = []
        correct = 0
        encoder = self.model[0]
        metric_module = self.model[1]
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            x = torch.cat([sample.squeeze() for sample in torch.split(x, 1)])
            target = target.flatten()
            # 拆分样本
            batch = x.shape[0]
            sample_1, sample_2 = torch.split(x, 1, dim=1)
            sample_1, sample_2 = sample_1.squeeze(1), sample_2.squeeze(1)
            x = torch.cat([sample_1, sample_2], dim=0)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            with torch.no_grad():
                h = encoder(x)
                h = h.reshape((h.shape[0], -1))
                sample_1, sample_2 = torch.split(h, batch, dim=0)
                pairs = torch.cat([sample_1, sample_2], dim=-1)
                similarity = metric_module(pairs)
            # 计算准确率
            pred = similarity.argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            # 计算crossentropy
            loss = criterion(similarity, target)  # 计算损失函数值
            # 计算
            losses.append(loss.item())
        return np.mean(losses), correct / (2 * len(data_loader.dataset))

    def get_parameters(self):
        return self.model.parameters()


class ClassTrainer(object):
    r"""
    Matching模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 训练过程
    def train(self, data_loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device, monitor=None):
        self.model.train()
        self.model.to(device)
        # encoder = self.model[0]
        # encoder.encoder.eval()
        # metric_module = self.model[1]
        criterion.to(device)
        losses = []
        grads = []
        for step, (x, target) in enumerate(data_loader):
            x, target = x.to(device), target.to(device)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            logits = self.model(x)
            # 计算crossentropy
            loss = criterion(logits, target)  # 计算损失函数值
            # 反向传播
            optimizer.zero_grad()
            # # 梯度裁剪
            # utils.clip_grad_value_(self.get_parameters(), 0.00001)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # 监控器
            if monitor is not None:
                grads.append(monitor(self.get_parameters()))
            if (step+1) % 5 == 0:
                print('batch:{} loss:{:.6f}'.format(step, loss.item()))
            losses.append(loss.item())
        if len(grads) > 0:
            grads_mean = np.mean(grads)
        else:
            grads_mean = None
        return np.mean(losses), grads_mean

    # 验证过程
    def evaluate(self, data_loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        self.model.to(device)
        criterion.to(device)
        losses = []
        correct = 0
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            with torch.no_grad():
                logits = self.model(x)
            # 计算准确率
            pred = logits.argmax(-1)
            correct += pred.eq(target).sum().item()
            # 计算crossentropy
            loss = criterion(logits, target)  # 计算损失函数值
            # 计算
            losses.append(loss.item())
        return np.mean(losses), correct / (2 * len(data_loader.dataset))

    def get_parameters(self):
        return self.model.parameters()


class PairTrainerNaive(object):
    r"""
    Matching模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 训练过程
    def train(self, data_loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device, monitor=None):
        self.model.train()
        self.model.to(device)
        encoder = self.model[0]
        encoder.encoder.eval()
        metric_module = self.model[1]
        criterion.to(device)
        losses = []
        grads = []
        for step, (x, target) in enumerate(data_loader):
            x, target = x.to(device), target.to(device)
            target = target.flatten()
            # 拆分样本
            batch = x.shape[0]
            sample_1, sample_2 = torch.split(x, 1, dim=1)
            sample_1, sample_2 = sample_1.squeeze(1), sample_2.squeeze(1)
            x = torch.cat([sample_1, sample_2], dim=0)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            h = encoder(x)
            h = h.reshape((h.shape[0], -1))
            sample_1, sample_2 = torch.split(h, batch, dim=0)
            pairs = torch.cat([sample_1, sample_2], dim=-1)
            similarity = metric_module(pairs)
            # 计算crossentropy
            loss = criterion(similarity, target)  # 计算损失函数值
            # 反向传播
            optimizer.zero_grad()
            # # 梯度裁剪
            # utils.clip_grad_value_(self.get_parameters(), 0.00001)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # 监控器
            if monitor is not None:
                grads.append(monitor(self.get_parameters()))
            if (step+1) % 5 == 0:
                print('batch:{} loss:{:.6f}'.format(step, loss.item()))
            losses.append(loss.item())
        if len(grads) > 0:
            grads_mean = np.mean(grads)
        else:
            grads_mean = None
        return np.mean(losses), grads_mean

    # 验证过程
    def evaluate(self, data_loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        self.model.to(device)
        criterion.to(device)
        losses = []
        correct = 0
        encoder = self.model[0]
        metric_module = self.model[1]
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            target = target.flatten()
            # 拆分样本
            batch = x.shape[0]
            sample_1, sample_2 = torch.split(x, 1, dim=1)
            sample_1, sample_2 = sample_1.squeeze(1), sample_2.squeeze(1)
            x = torch.cat([sample_1, sample_2], dim=0)
            # 维度变化
            x = x.permute((0, 3, 1, 2)).unsqueeze(1)
            with torch.no_grad():
                h = encoder(x)
                h = h.reshape((h.shape[0], -1))
                sample_1, sample_2 = torch.split(h, batch, dim=0)
                pairs = torch.cat([sample_1, sample_2], dim=-1)
                similarity = metric_module(pairs)
            # 计算准确率
            pred = torch.argmax(similarity, dim=1)
            correct += pred.eq(target).sum().item()
            # 计算crossentropy
            loss = criterion(similarity, target)  # 计算损失函数值
            # 计算
            losses.append(loss.item())
        return np.mean(losses), correct / len(data_loader.dataset)

    def get_parameters(self):
        return self.model.parameters()
