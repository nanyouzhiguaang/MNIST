import argparse
import math
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import mnist_loader as ml
from torchvision import transforms
from models.cnn import Net


def train():
    os.makedirs('./output', exist_ok=True)                                          # 新建一个目录存放训练所需要输出的内容
    if True:
        ml.image_list(args.datapath, 'output/total.txt')                            # 根据训练数据的目录来输出一个图像列表的文件
        ml.shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')  # 把图像列表文件分成训练集和验证集

    train_data = ml.MyDataset(txt='output/train.txt', transform=transforms.ToTensor())  # 预处理采用ToTensor的方法
                                                                                        # 将像素值从0-255归一化到0-1之间
    val_data = ml.MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)  # 做好数据集之后需要将数据一个批次一个批次地加载进去
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)  # batch_size表示一次加载多少个数据

    model = Net()

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # 优化器采用Adam这种方法
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20],
                                                     0.1)  # 当达到第10个epoch时lr乘以0.1，当达到第20个epoch再乘以0.1
    loss_func = nn.CrossEntropyLoss()  # 定义损失函数采用交叉熵损失函数

    # training-------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):  # batch_x 表示图片 batch_y 表示标签
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256*3*28*28   out 256*10（256代表每次输出256张图，10代表每张图片输出10个值）
            loss = loss_func(out, batch_y)  # 将 out结果和标签做对比计算损失函数
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]  # 1表示按行取最大值 [1] 返回最大值的索引
            train_correct = (pred == batch_y).sum()  # 预测结果和标签做比较，算出一个批次做对多少个
            train_acc += train_correct.item()
            print('epoch:%2d/%d  batch:%3d/%d  TrainLoss:%.3f.  Acc:%.3f'
                  % (epoch + 1, args.epochs, batch, math.ceil(len(train_data) / args.batch_size),
                     loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()  # 清空过往梯度；
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 更新网络参数
            # 总结来说：梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，不断累加，累加一定次数后，
            # 根据累加的梯度更新网络参数，然后清空梯度，进行下一次循环。
        scheduler.step()  # 更新lr
        print('TrainLoss:%.6f, Acc: %.3f' % (train_loss / (math.ceil(len(train_data) / args.batch_size)),
                                             train_acc / (len(train_data))))

        # evaluation----------做完一个epoch 就要做一次评估
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss:%.6f, Acc:%.3f' % (eval_loss / (math.ceil(len(val_data) / args.batch_size)),
                                           eval_acc / (len(val_data))))

        # save model-------------------------------------------------------
        if (epoch + 1) % 3 == 0:                       # 可以每个epoch都保存模型，也可以隔几个epoch再保存
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--datapath', required=True, help='data path')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--use_cuda', default=False, help='using CUDA for training')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    train()
