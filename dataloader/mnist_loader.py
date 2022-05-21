import os
import random

import cv2
from torch.utils.data import Dataset


def image_list(imageRoot, txt='list.txt'):
    f = open(txt, 'wt')  # wt表示以文本方式打开
    for (label, filename) in enumerate(
            sorted(os.listdir(imageRoot), reverse=False)):    # enumerate 是枚举 os.listdir()用于返回绝对路径
        if os.path.isdir(os.path.join(imageRoot, filename)):  # os.path.isdir() 用于判断路径是否为目录
            for imagename in os.listdir(os.path.join(imageRoot, filename)):
                name, ext = os.path.splitext(imagename)  # 分离图片名与扩展名(.jpg)
                ext = ext[1:]  # ext为扩展名
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n' % (os.path.join(imageRoot, filename, imagename), label))  # label是哪里来的
    f.close()


def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:  # ===>f = open(listFile, 'r')
        records = f.readlines()  # 从文件读取一行数据
    random.shuffle(records)  # 将列表的元素顺序打乱
    num = len(records)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as f:
        f.writelines(records[0:trainNum])  # 向指定的文件写入数据
    with open(valFile, 'w') as f1:
        f1.writelines(records[trainNum:])


class MyDataset(Dataset):                             # 继承 pytorch 的Dataset类  这个类的目的就是传入数据的路径，和预处理部分（看参数），然后给我们返回数据，
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()                # 用来去除结尾字符
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform              # 将数据转化为tensor
        self.target_transform = target_transform

    def __getitem__(self, index):               # 返回 img 和label img已经tensor化了
        fn, label = self.imgs[index]           # fn表示图片的绝对路径
        img = cv2.imread(fn, cv2.IMREAD_COLOR)   # cv2.imread()读取图片后以多维数组的形式保存图片信息，
                                                 # 前两维表示图片的像素坐标，最后一维表示图片的通道索引
        # 将数据转化为tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):                          # 返回训练集数据的大小
        return len(self.imgs)
