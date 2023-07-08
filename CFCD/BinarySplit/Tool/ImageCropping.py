import cv2
import os
import math
import numpy as np
import random


class ImgCrop(object):
    def __init__(self, DataImage, DataLabel, WinSize, ClaNum, DataSavePath, Per_sample):
        self.DataImage = DataImage
        self.DataLabel = DataLabel
        self.WinSize = WinSize
        self.ClaNum = ClaNum
        self.Per_sample = Per_sample
        self.DataSavePath = DataSavePath

    # 将图像裁剪成块
    def WinDoImg(self, PixelPois, Train=False, label=0):
        if Train == True:  # 裁剪训练集和验证集数据
            ImageBlock = np.zeros([self.WinSize, self.WinSize, 4], dtype=np.uint8)  # 创建4维图像
            MinX = PixelPois[0] - math.floor(0.5 * (self.WinSize + 1))
            MinY = PixelPois[1] - math.floor(0.5 * (self.WinSize + 1))
            MaxX = PixelPois[0] + math.floor(0.5 * (self.WinSize - 1))
            MaxY = PixelPois[1] + math.floor(0.5 * (self.WinSize - 1))
            x = 0  # 为新创建的图像按照横坐标赋值
            for i in range(MinX, MaxX + 1):
                y = 0  # 为新创建的图像按照纵坐标赋值
                for j in range(MinY, MaxY + 1):
                    if (0 <= i and i < (self.DataImage.shape[0] - 1)) and (0 <= j and j < self.DataImage.shape[1]):
                        ImageBlock[x, y, 0] = self.DataImage[i, j, 0]  # B通道赋值
                        ImageBlock[x, y, 1] = self.DataImage[i, j, 1]  # G通道赋值
                        ImageBlock[x, y, 2] = self.DataImage[i, j, 2]  # R通道赋值
                        ImageBlock[x, y, 3] = label  # 赋值标签
                    else:
                        ImageBlock[x, y, 0] = 0
                        ImageBlock[x, y, 1] = 0
                        ImageBlock[x, y, 2] = 0
                        ImageBlock[x, y, 3] = label  # 赋值标签
                    y = y + 1
                x = x + 1
        else:  # 裁剪测试集数据
            ImageBlock = np.zeros([self.WinSize, self.WinSize, 3], dtype=np.uint8)  # 创建3维图像
            MinX = PixelPois[0] - math.floor(0.5 * (self.WinSize + 1))
            MinY = PixelPois[1] - math.floor(0.5 * (self.WinSize + 1))
            MaxX = PixelPois[0] + math.floor(0.5 * (self.WinSize - 1))
            MaxY = PixelPois[1] + math.floor(0.5 * (self.WinSize - 1))
            x = 0  # 为新创建的图像按照横坐标赋值
            for i in range(MinX, MaxX + 1):
                y = 0  # 为新创建的图像按照纵坐标赋值
                for j in range(MinY, MaxY + 1):
                    if (0 <= i and i < (self.DataImage.shape[0] - 1)) and (0 <= j and j < self.DataImage.shape[1]):
                        ImageBlock[x, y, 0] = self.DataImage[i, j, 0]  # B通道赋值
                        ImageBlock[x, y, 1] = self.DataImage[i, j, 1]  # G通道赋值
                        ImageBlock[x, y, 2] = self.DataImage[i, j, 2]  # R通道赋值
                    else:
                        ImageBlock[x, y, 0] = 0
                        ImageBlock[x, y, 1] = 0
                        ImageBlock[x, y, 2] = 0
                    y = y + 1
                x = x + 1

        return ImageBlock

    def TestandTrain(self):
        """遍历并裁剪图像"""
        # 存储验证集数据
        TrainingXY = [[] for i in range(self.ClaNum + 1)]  # 按类别保存每一类的训练样本的路径
        ValidSavePath = self.DataSavePath + '\\ValidSet'  # 初始化验证集路径
        if not os.path.exists(ValidSavePath):
            os.mkdir(ValidSavePath)
        for row in range(self.DataImage.shape[0]):
            for col in range(self.DataImage.shape[1]):
                TupPixel = (row, col)
                label = self.DataLabel[TupPixel[0], TupPixel[1], 0]  # 获取像素的标签
                if label != 0:
                    DoImage = ImgCrop.WinDoImg(self, TupPixel, Train=True, label=label)  # 裁剪数据
                    ValidImagePath = ValidSavePath + '\\%d-%d-%d' % (label, row, col) + '.png'
                    cv2.imwrite(ValidImagePath, DoImage)  # 按照格式保存裁剪后的图像
                    TrainingXY[label].append(ValidImagePath)
        # 存储训练集数据
        TrainSavePath = self.DataSavePath + '\\Train'  # 初始化训练集路径
        TempTrainSavePath = []  # 保存训练集的路径
        if not os.path.exists(TrainSavePath):
            os.mkdir(TrainSavePath)
        # 随机选取训练样本
        for i in range(1, self.ClaNum + 1):
            RandNum = random.sample(range(0, len(TrainingXY[i]) - 1), Per_sample)
            for S in range(len(RandNum)):
                labelImages = cv2.imread(TrainingXY[i][RandNum[S]], flags=-1)  # 读取出来深度保持不变
                TrainImagePath = TrainingXY[i][RandNum[S]].replace('ValidSet', 'Train')  # 将图像写作训练集路径下
                TestPath = TrainingXY[i][RandNum[S]].replace('ValidSet', 'Test')
                TempTrainSavePath.append(TestPath)  # 将图像写作训练集路径下
                cv2.imwrite(TrainImagePath, labelImages)
        # 存储测试集数据
        TestImagePath = self.DataSavePath + '\\Test'  # 初始化所有数据路径
        if not os.path.exists(TestImagePath):
            os.mkdir(TestImagePath)
        for row in range(self.DataImage.shape[0]):
            for col in range(self.DataImage.shape[1]):
                TupPixel = (row, col)
                DoImage = ImgCrop.WinDoImg(self, TupPixel)  # 裁剪数据
                label = self.DataLabel[TupPixel[0], TupPixel[1], 0]  # 获取像素的标签
                if label != 0:
                    TestSavePath = TestImagePath + '\\%d-%d-%d' % (label, row, col) + '.png'  # 存储测试数据
                    for TempPath in range(len(TempTrainSavePath)):
                        if TempTrainSavePath[TempPath] == TestSavePath:
                            break
                        elif TempPath == len(TempTrainSavePath) - 1:
                            TestSavePath = TestImagePath + '\\unlabel-%d-%d' % (row, col) + '.png'  # 存储测试数据
                            cv2.imwrite(TestSavePath, DoImage)  # 按照格式保存裁剪后的图像 """
                else:
                    TestSavePath = TestImagePath + '\\unlabel-%d-%d' % (row, col) + '.png'
                    cv2.imwrite(TestSavePath, DoImage)  # 按照格式保存裁剪后的图像 """


if __name__ == '__main__':
    # 定义参数
    DataImage = cv2.imread('D:\\Classification\\Paper_03_20220323\\Data\\Salinas\\Salinas_PCA1.tif')
    DataLabel = cv2.imread('D:\\Classification\\Paper_03_20220323\\Data\\Salinas\\Salinas_GT.png')
    DataSavePath = 'D:\\Classification\\Paper_03_20220323\\Experiment-4\\PCA1'  # 裁剪后图片保存的主干路径
    WinSize = 32  # 裁剪图片的大小设置
    ClaNum = 16  # 数据类别数量
    Per_sample = 15  # 每类取多少训练样本
    random.seed(9)  # 设置随机数种子
    # 加载训练数据
    ImgCrops = ImgCrop(DataImage, DataLabel, WinSize, ClaNum, DataSavePath, Per_sample)
    ImgCrop.TestandTrain(ImgCrops)
