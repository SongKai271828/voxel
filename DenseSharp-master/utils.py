import torch as pt
import numpy as np
from matplotlib import pyplot as plt
import random
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


# 观察三维数据的切片图函数
def data_visualize(path):
    tmp = np.load(path)
    voxels = tmp['voxel']
    segs = tmp['seg']
    up = 60
    down = 40
    interval = 1

    for i in range(down, up, interval):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(voxels[i])
        plt.subplot(1, 2, 2)
        plt.imshow(voxels[i]*(segs[i].astype(np.float)))
        plt.show()


# 数据集类，用于导入数据和数据增强
class Dataset:
    def __init__(self, data_path, label_path, batch, type, pre, deal_size, enhance, intep=False):
        self.data_path = data_path	# 数据路径
        self.label_path = label_path	# 标签路径
        self.batch = batch		# batch size
        self.preprocess = pre		# 是否进行预处理(截取)
        self.deal_size = deal_size	# 实际截取的尺寸
        self.enhance = enhance		# 是否需要数据增强
        self.intep = intep		# 是否采用内插插值

        self.real_size = 960		# 使用数据增强时的实际数据集大小
        self.channel_in = 1		# 输入通道的个数
        self.enhance_type = [0, 1]	# 选择数据增强的类型，第0位为mix up，第1位为翻转

        import pandas as pd
        self.base = pd.read_csv(self.label_path)
        self.names = np.array(self.base['name'])
        if type == 'train':
            self.labels = np.array(self.base['lable'])
        else:
            self.labels = np.array(self.base['lable'])
        self.size = np.shape(self.names)[0]
    
    # 数据集求大小，数据增强与不进行数据增强时数据集大小不同
    def __len__(self):
        if self.enhance:
            return self.real_size
        else:
            return self.size
    
    # 求出截取的位置，非内插时只截取中心deal_size的部分，内插时求出结界的实际边界，
    def get_edge(self, voxels):
        size = np.shape(voxels)
        zeros = np.zeros([size[0], size[1]])
        min = []
        max = []
        border = 0
        # 寻找结节的实际边界
        for i in range(size[0]):
            if (voxels[i, :, :] != zeros).any():
                break
        min.append(i)
        for i in range(size[0]):
            if (voxels[size[0] - 1 - i, :, :] != zeros).any():
                break
        max.append(size[0] - 1 - i)

        for i in range(size[1]):
            if (voxels[:, i, :] != zeros).any():
                break
        min.append(i)
        for i in range(size[1]):
            if (voxels[:, size[1] - 1 - i, :] != zeros).any():
                break
        max.append(size[1] - 1 - i)

        for i in range(size[2]):
            if (voxels[:, :, i] != zeros).any():
                break
        min.append(i)
        for i in range(size[2]):
            if (voxels[:, :, size[2] - 1 - i] != zeros).any():
                break
        max.append(size[2] - 1 - i)


        for i in range(len(min)):
            if self.intep:
                min[i] -= border
            else:
                min[i] = np.shape(voxels)[0]//2-self.deal_size//2 + border
        for i in range(len(max)):
            if self.intep:
                max[i] += border
            else:
                max[i] = min[i]+self.deal_size-1 - border
        return min, max
    
    # 截取数据与内插
    def change_size(self, voxels, segs, new_size):
        min, max = self.get_edge(segs)
        new = voxels[min[0]:max[0] + 1, min[1]:max[1] + 1, min[2]:max[2] + 1]
        new_segs = segs[min[0]:max[0] + 1, min[1]:max[1] + 1, min[2]:max[2] + 1]

        new_segs = new_segs.astype(np.float)
        new_voxels = new.astype(np.float)
        # 是否进行内插
        if self.intep:
            new_voxels = pt.tensor([[new_voxels]])
            new_segs = pt.tensor([[new_segs]])

            new_voxels = pt.nn.functional.interpolate(new_voxels, size=new_size, mode='trilinear', align_corners=False)
            new_segs = pt.nn.functional.interpolate(new_segs, size=new_size, mode='trilinear', align_corners=False)

            new_segs = pt.round(new_segs)
            new_voxels = new_voxels * new_segs
            new_voxels = np.array(new_voxels[0][0])
        else:
            new_voxels = new_voxels * (new_segs)
        return np.array([new_voxels])
    
    # 操作符重载，获得单个数据
    def __getitem__(self, item):
        # 不需要数据增强
        if not self.enhance:
            data = np.load(self.data_path+self.names[item]+'.npz')
            label = self.labels[item]
            label = np.array([1-label, label], dtype=np.float)
            if self.preprocess:
                data = self.change_size(data['voxel'], data['seg'], self.deal_size)
            else:
                low = np.shape(data)[0] // 2 - self.deal_size // 2
                high = low + self.deal_size
                data = data[low:high, low:high, low:high]
                data = np.array([data])
        # 需要数据增强
        else:
            if item < self.size:
                data = np.load(self.data_path + self.names[item] + '.npz')
                label = self.labels[item]
                label = np.array([1-label, label], dtype=np.float)
                if self.preprocess:
                    data = self.change_size(data['voxel'], data['seg'], self.deal_size)
                else:
                    low = np.shape(data)[0] // 2 - self.deal_size // 2
                    high = low + self.deal_size
                    data = data[low:high, low:high, low:high]
                    data = np.array([data])
            else:
                if self.enhance_type[0]:
                    data, label = self.mix()
                if self.enhance_type[1]:
                    data, label = self.mirror()

        # print(data.size())
        return data.astype(dtype=np.float), label

    # mix up数据增强
    def mix(self):
        place1 = random.randint(0, self.size - 1)
        place2 = random.randint(0, self.size - 1)
        data1 = np.load(self.data_path + self.names[place1] + '.npz')
        label1 = pt.tensor([1-self.labels[place1], self.labels[place1]])
        data2 = np.load(self.data_path + self.names[place2] + '.npz')
        label2 = pt.tensor([1-self.labels[place2], self.labels[place2]])
        # lam = random.random()
        lam = 0.4

        if self.preprocess:
            data1 = self.change_size(data1['voxel'], data1['seg'], self.deal_size)
            data2 = self.change_size(data2['voxel'], data2['seg'], self.deal_size)
        else:
            low = np.shape(data1)[0] // 2 - self.deal_size // 2
            high = low + self.deal_size
            data1 = data1[low:high, low:high, low:high]
            data1 = np.array([data1])
            data2 = data2[low:high, low:high, low:high]
            data2 = np.array([data2])
        data = lam * data1 + (1 - lam) * data2
        label = lam * float(label1[1]) + (1 - lam) * float(label2[1])
        label = np.array([label, 1-label], dtype=np.float)
        # label = int((label + 0.49) // 1)
        return data, label

    # 数据镜像翻转
    def mirror(self):
        place1 = random.randint(0, self.size - 1)
        data1 = np.load(self.data_path + self.names[place1] + '.npz')
        label1 = np.array([1-self.labels[place1], self.labels[place1]], dtype=np.float)

        if self.preprocess:
            data1 = self.change_size(data1['voxel'], data1['seg'], self.deal_size)
        else:
            low = np.shape(data1)[0] // 2 - self.deal_size // 2
            high = low + self.deal_size
            data1 = data1[low:high, low:high, low:high]
            data1 = np.array([data1])

        dims = [1, 2, 3]
        random.shuffle(dims)
        data2 = data1.transpose([0] + dims)
        # 查看反转结果
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(data1[0, :, :, 16]))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(data2[0, :, :, 16]))
        # plt.show()

        return data2, label1

    # 改变是否需要数据增强
    def change_mode(self, mode):
        self.enhance = mode

    def get_batch_size(self):
        return self.batch

    # 一次性生成全部数据
    def load_all(self):
        size = len(self)
        all_data = np.zeros([size, self.deal_size, self.deal_size, self.deal_size, self.channel_in])
        all_label = np.zeros([size, 2])

        for i in range(size):
            tmp_data, tmp_label = self.__getitem__(i)
            tmp_data = tmp_data[0, :, :, :, np.newaxis]

            # 查看数据生成情况
            # plt.figure()
            # plt.imshow(np.array(tmp_data[:, :, 16, 0]))
            # plt.show()

            all_data[i] = np.array(tmp_data, dtype=np.float)
            all_label[i] = np.array(tmp_label, dtype=np.float)

        return all_data, all_label


# 基于pytorch的DenseNet定义
class ConvBlock(nn.Sequential):
    def __init__(self, channel_in, gain, bn_size):
        super(ConvBlock, self).__init__()
        self.add_module('BN1', nn.BatchNorm3d(channel_in))
        self.add_module('Relu1', nn.ReLU(inplace=True))
        self.add_module('Conv3D1', nn.Conv3d(channel_in, bn_size * gain,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1, bias=True))
        self.add_module('Dropout1', nn.Dropout3d(p=0.1))
        self.add_module('BN2', nn.BatchNorm3d(bn_size*gain))
        self.add_module('Relu2', nn.ReLU(inplace=True))
        self.add_module('Conv3D2', nn.Conv3d(bn_size*gain, gain,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=True))

    def forward(self, x):
        features = super(ConvBlock, self).forward(x)
        return pt.cat([x, features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, layer_num, channel_in, bn_size, gain):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            self.add_module('ConvBlock%d' % (i+1),
                            ConvBlock(channel_in + gain * i,
                                      gain, bn_size))
            self.add_module('Dropout', nn.Dropout3d(p=0.1))


class TransitionBlock(nn.Sequential):
    def __init__(self, channel_in, channel_out):
        super(TransitionBlock, self).__init__()
        self.add_module('BN', nn.BatchNorm3d(channel_in))
        self.add_module('Relu', nn.ReLU(inplace=True))
        self.add_module('Conv3D', nn.Conv3d(channel_in, channel_out,
                                          kernel_size=1,
                                          stride=1, bias=True))
        self.add_module('Pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, gain=12, block_set=(6, 12, 24, 16),
                 bn_size=4, feature_compress=0.5, types=10):
        super(DenseNet, self).__init__()

        init_feature_num = 2 * gain

        self.features_gain = nn.Sequential(OrderedDict([
            ('Conv3D0', nn.Conv3d(1, init_feature_num,
                                kernel_size=7, stride=2,
                                padding=3, bias=True)),
            ('BN0', nn.BatchNorm3d(init_feature_num)),
            ('Relu0', nn.ReLU(inplace=True)),
            ('Pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        ]))



        features_num = init_feature_num
        for i, num_layers in enumerate(block_set):
            self.features_gain.add_module('DenseBlock%d' % (i + 1),
                                          DenseBlock(num_layers, features_num,
                                                bn_size, gain))
            features_num = features_num + gain * num_layers
            if i != len(block_set)-1:
                self.features_gain.add_module('TransitionBlock%d' % (i + 1),
                                              TransitionBlock(features_num,
                                                         int(features_num * feature_compress)))
                features_num = int(features_num * feature_compress)

        self.features_gain.add_module('Norm', nn.BatchNorm3d(features_num))
        self.features_gain.add_module('Relu', nn.ReLU(inplace=True))
        self.features_gain.add_module('AvgPool', nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.FullyConnector = nn.Linear(features_num, types)

    def forward(self, x):
        features = self.features_gain(x)
        out = features.view(features.size(0), -1)
        out = self.FullyConnector(out)
        out = F.softmax(out, dim=1)
        return out


# 生城pytorch的卷积层
def conv(dim, in_c, out_c, kernal, pad=0, stri=1):
    if dim == 1:
        return pt.nn.Conv1d(in_channels=in_c,
                            out_channels=out_c,
                            kernel_size=kernal,
                            stride=stri,
                            padding=pad,
                            bias=True)
    elif dim == 2:
        return pt.nn.Conv2d(in_channels=in_c,
                            out_channels=out_c,
                            kernel_size=(kernal, kernal),
                            stride=(stri, stri),
                            padding=pad,
                            bias=True)
    else:
        return pt.nn.Conv3d(in_channels=in_c,
                            out_channels=out_c,
                            kernel_size=(kernal, kernal, kernal),
                            stride=(stri, stri, stri),
                            padding=pad,
                            bias=True)


# 生成pytorch的全连接层
def line(in_f, out_f):
    return pt.nn.Linear(in_features=in_f,
                        out_features=out_f,
                        bias=True)


# 生成pytorch的池化层
def pool(dim, type, kernal, pad=0):
    if dim == 3:
        if type == 'max':
            return pt.nn.MaxPool3d(kernel_size=kernal,
                                   stride=1,
                                   padding=pad)
        else:
            return pt.nn.AvgPool3d(kernel_size=kernal,
                                   stride=1,
                                   padding=pad)
    elif dim == 1:
        if type == 'max':
            return pt.nn.MaxPool1d(kernel_size=kernal,
                                   padding=pad)
        else:
            return pt.nn.AvgPool1d(kernel_size=kernal,
                                   padding=pad)
    else:
        if type == 'max':
            return pt.nn.MaxPool2d(kernel_size=kernal,
                                   padding=pad)
        else:
            return pt.nn.AvgPool2d(kernel_size=kernal,
                                   padding=pad)


# 生成pytorch的batchnorm层
def norm(dim, in_f):
    if dim == 3:
        return pt.nn.BatchNorm3d(num_features=in_f)
    elif dim == 2:
        return pt.nn.BatchNorm2d(num_features=in_f)
    else:
        return pt.nn.BatchNorm1d(num_features=in_f)


# 自定义的损失函数
def loss(pred, label):
    beta = 1.
    smooth = 1.
    bb = beta*beta
    intersection = pt.sum(pred*label)
    weight_union = bb * pt.sum(label) + pt.sum(pred)
    score = -((1+bb) * intersection + smooth) / (weight_union + smooth)
    return score


if __name__ == '__main__':
    from Global import *

    BATCH_SIZE = 20
    DEAL_SIZE = 32

    train_set = Dataset(
        data_path=train_path,
        label_path=train_label_path,
        batch=BATCH_SIZE,
        type='train',
        pre=True,
        deal_size=DEAL_SIZE,
        enhance=True,
        # intep=True
        intep=False
    )
    train_set.load_all()
