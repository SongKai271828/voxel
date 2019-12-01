import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import importlib
import torch as pt
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

TRAIN_SET = 465
TEST_SET = 118
DATA_SIZE = [100, 100, 100]

#DEVICE = pt.device("cuda" if pt.cuda.is_available() else "cpu")
DEVICE = pt.device("cpu")
EPOCH = 20
TRAIN_SIZE = 20
TEST_SIZE = 20
BATCH_SIZE = 4
TRAIN_NUM = TRAIN_SIZE//BATCH_SIZE
TEST_NUM = TEST_SIZE//BATCH_SIZE
CHANNEL_IN = 1
DEAL_SIZE = [100, 100, 100]
TYPE_NUM = 2

test_label_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/sampleSubmission.csv'
test_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/data_test/'
train_label_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/train_val.csv'
train_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/data_train/'
# sys.path.append(test_path)
# sys.path.append(train_path)
test_label = pd.read_csv(test_label_path)
train_label = pd.read_csv(train_label_path)
sys.path.append(test_label_path)
sys.path.append(train_label_path)


def get_dataset(device='cpu'):
    train_names = np.array(train_label['name'])
    test_names = np.array(test_label['name'])
    train_scores = np.array(train_label['lable'])
    test_scores = np.array(test_label['Score'])

    train_scores = pt.tensor(train_scores)
    test_scores = pt.tensor(test_scores)

    if device == 'cpu':
        train_voxels = pt.FloatTensor(TRAIN_NUM, BATCH_SIZE, CHANNEL_IN, 100, 100, 100).zero_()
        train_segs = pt.FloatTensor(TRAIN_NUM, BATCH_SIZE, CHANNEL_IN, 100, 100, 100).zero_()
        #train_targets = pt.FloatTensor(TRAIN_NUM, BATCH_SIZE, TYPE_NUM).zero_()
        train_targets = pt.LongTensor(TRAIN_NUM, BATCH_SIZE).zero_()

        test_voxels = pt.FloatTensor(TEST_NUM, BATCH_SIZE,CHANNEL_IN, 100, 100, 100).zero_()
        test_segs = pt.FloatTensor(TEST_NUM, BATCH_SIZE,CHANNEL_IN, 100, 100, 100).zero_()
        test_targets = pt.FloatTensor(TEST_NUM, BATCH_SIZE, TYPE_NUM).zero_()
        #test_targets = pt.FloatTensor(TEST_NUM, BATCH_SIZE).zero_()
    else:
        train_voxels = pt.cuda.FloatTensor(TRAIN_NUM, BATCH_SIZE, CHANNEL_IN, 100, 100, 100)
        train_segs = pt.cuda.FloatTensor(TRAIN_NUM, BATCH_SIZE, CHANNEL_IN, 100, 100, 100)
        train_targets = pt.cuda.LongTensor(TRAIN_NUM, BATCH_SIZE)

        test_voxels = pt.cuda.FloatTensor(TEST_NUM, BATCH_SIZE, CHANNEL_IN, 100, 100, 100)
        test_segs = pt.cuda.FloatTensor(TEST_NUM, BATCH_SIZE, CHANNEL_IN, 100, 100, 100)
        test_targets = pt.cuda.FloatTensor(TEST_NUM, BATCH_SIZE, TYPE_NUM).zero_()

    for i in range(TRAIN_SIZE):
        tmp = np.load(train_path+train_names[i]+'.npz')
        row = int(np.floor(i / BATCH_SIZE))
        col = i % BATCH_SIZE
        #print(row, col)

        train_voxels[row, col] = pt.tensor(tmp['voxel'])
        train_segs[row, col] = pt.tensor(tmp['seg'])
        #train_targets[row, col] = pt.tensor([float(train_scores[i]), 1-float(train_scores[i])+0.0001])
        train_targets[row, col] = train_scores[i]

    for i in range(TEST_SIZE):
        tmp = np.load(test_path+test_names[i]+'.npz')
        row = int(np.floor(i / BATCH_SIZE))
        col = i % BATCH_SIZE
        #print(row, col)

        test_voxels[row, col] = pt.tensor(tmp['voxel'])
        test_segs[row, col] = pt.tensor(tmp['seg'])
        test_targets[row, col] = pt.tensor([test_scores[i], 1-test_scores[i]+0.0001])

    return train_voxels, train_segs, train_targets, test_voxels, test_segs, test_targets


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_64 = pt.nn.Conv3d(in_channels=1,
                                    out_channels=64,
                                    kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1),
                                    bias=True)
        self.conv64_32 = pt.nn.Conv3d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1),
                                    bias=True)
        self.conv32_16 = pt.nn.Conv3d(in_channels=32,
                                    out_channels=16,
                                    kernel_size=(3, 3, 3),
                                    stride=(2, 2, 2),
                                    bias=True)
        self.conv16_1 = pt.nn.Conv3d(in_channels=16,
                                    out_channels=1,
                                    kernel_size=(1, 1, 1),
                                    bias=True)
        self.conv64_128 = pt.nn.Conv3d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(3, 3, 3),
                                    stride=(2, 2, 2),
                                    bias=True)
        self.conv128_64 = pt.nn.Conv3d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=(1, 1, 1),
                                    bias=True)
        self.maxpool8 = pt.nn.MaxPool3d(kernel_size=(8, 8, 8))
        self.maxpool4 = pt.nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.maxpool3 = pt.nn.MaxPool3d(kernel_size=(3, 3, 3))
        self.avgpool8 = pt.nn.AvgPool3d(kernel_size=(8, 8, 8))
        self.avgpool4 = pt.nn.AvgPool3d(kernel_size=(4, 4, 4))
        self.avgpool3 = pt.nn.AvgPool3d(kernel_size=(3, 3, 3))

        self.batchnorm64a = pt.nn.BatchNorm3d(num_features=64)
        self.batchnorm64b = pt.nn.BatchNorm3d(num_features=64)
        self.batchnorm32 = pt.nn.BatchNorm3d(num_features=32)
        self.batchnorm16 = pt.nn.BatchNorm3d(num_features=16)
        self.batchnorm128 = pt.nn.BatchNorm3d(num_features=128)

        # self.conv3d1 = pt.nn.Conv2d(in_channels=100,
        #                             out_channels=64,
        #                             kernel_size=5,
        #                             stride=5,
        #                             bias=True)
        # self.conv3d2 = pt.nn.Conv2d(in_channels=64,
        #                             out_channels=32,
        #                             kernel_size=(3, 3),
        #                             stride=(2, 2),
        #                             bias=True)
        # self.conv3d3 = pt.nn.Conv2d(in_channels=32,
        #                             out_channels=16,
        #                             kernel_size=(3, 3),
        #                             stride=(2, 2),
        #                             bias=True)
        # self.conv3d4 = pt.nn.Conv2d(in_channels=16,
        #                             out_channels=1,
        #                             kernel_size=(3, 3),
        #                             bias=True)
        # self.maxpool1 = pt.nn.MaxPool2d(kernel_size=(8, 8))
        # self.maxpool2 = pt.nn.MaxPool2d(kernel_size=(4, 4))
        # self.maxpool3 = pt.nn.MaxPool2d(kernel_size=(3, 3))
        # self.avgpool1 = pt.nn.AvgPool2d(kernel_size=(8, 8))
        # self.avgpool2 = pt.nn.AvgPool2d(kernel_size=(4, 4))

        #self.linear1 = pt.nn.Linear(in_features=1)

        self.linear1 = pt.nn.Linear(in_features=32*4*4*4,
                                    out_features=32*8*8*8,
                                    bias=True)
        self.linear2 = pt.nn.Linear(in_features=16*3*3*3,
                                    out_features=4*2*2*2,
                                    bias=True)
        self.linear3 = pt.nn.Linear(in_features=4*2*2*2,
                                    out_features=2,
                                    bias=True)
        self.linear4 = pt.nn.Linear(in_features=2*2*2*2,
                                    out_features=2)

    def forward(self, voxel):
        out = self.conv1_64(voxel)
        out = self.batchnorm64a(out)
        out = F.relu(out)

        out = self.conv64_128(out)
        out = self.batchnorm128(out)
        out - F.relu(out)

        out = self.maxpool4(out)
        #print(out.size(),1)

        out = F.relu(out)

        out = self.conv128_64(out)
        out = self.batchnorm64b(out)
        out = F.relu(out)

        out = self.avgpool3(out)
        #print(out.size(),2)

        out = self.conv64_32(out)
        out = F.relu(out)

        out = out.view(BATCH_SIZE, -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = out.view(BATCH_SIZE, 32, 8, 8, 8)

        out = self.conv32_16(out)
        out = self.batchnorm16(out)
        #print(out.size(),3)

        out = out.view(BATCH_SIZE, -1)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        # out = self.linear4(out)
        out = out.view(BATCH_SIZE, 2)

        out = F.softmax(out, dim=1)

        return out


model = Net()
model.to(DEVICE)   # 将网络移动到gpu上
optimizer = pt.optim.Adam(model.parameters())
cri = F.nll_loss


def train_and_eval(model, trv, trs, trt, tev, tes, tet, cri):
    for i in range(EPOCH):
        print('**************************************************')
        print('Epoch: ', i+1)
        model.train()
        print('Start train')
        for j in range(TRAIN_NUM):
            data = trv[j]
            label = trt[j]
            optimizer.zero_grad()
            predict = model(data)
            #print('predict: ', predict)
            #print('labels: ', label)
            loss = cri(predict, label)
            loss.backward()
            optimizer.step()

        model.eval()
        print('Start eval')
        distance = 0
        with pt.no_grad():
            print('---------TRAIN_DATASET--------')
            correct_num_train = 0
            for j in range(TRAIN_NUM):
                data = trv[j]
                label = trt[j]
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                output = model(data)
                predict = output.max(1, keepdim=True)[1]
                correct_num_train += predict.eq(label.view_as(predict)).sum().item()
                #print('Predict:{0:.6}, Truth:{1:.6}'.format(predict, label))
                #distance += (predict-label)*(predict-label)
            correct_rate = 100. * correct_num_train / TRAIN_SIZE
            print('Train Accuracy: {0}/{1}({2:.4}%)'.format(correct_num_train,TRAIN_SIZE,correct_rate))
            # print('Average distance: {0:6}').format(distance)
            # print('---------TEST_DATASET---------')
            # for j in range(TEST_SIZE):
            #     data = tev[j]
            #     label = tet[j]
            #     data = data.to(DEVICE)
            #     label = label.to(DEVICE)
            #     predict = model(data)
            #     #print('Predict:{0:.6}, Truth:{1:.6}'.format(predict, label))
            #     distance += (predict-label)*(predict-label)
            # distance /= tev.size()[1]
            # print('Average distance: {0:6}'.format(distance))


if __name__ == '__main__':
    trv, trs, trt, tev, tes, tet = get_dataset()
    train_and_eval(model, trv, trs, trt, tev, tes, tet, cri)
