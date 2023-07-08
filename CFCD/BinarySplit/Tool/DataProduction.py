import cv2
import torch
from torch.utils.data import Dataset
import random
import glob
import os
import numpy as np
import torchvision


class ISBI_Loader(Dataset):
    # 初始化函数，读取所有DataPathName下的图片
    def __init__(self, data_path, transform, data_type='no-Test'):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.data_type = data_type
        self.images_path = glob.glob(os.path.join('%s\\RGB\\*.png' % data_path))
        self.transform = transform

    def augment(self, TrainImage, flipCode):
        # 使用cv2.flip进行数据增强，fillipCode为1水平翻转，0为垂直翻转，-1水平翻转
        flip = cv2.flip(TrainImage, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图像
        if self.data_type == 'Test':
            rgb_path = self.images_path[index]
            pca1_path = rgb_path.replace("RGB", "PCA1")
            pca2_path = rgb_path.replace("RGB", "PCA2")

            rgb_image = cv2.imread(rgb_path)
            pca1_image = cv2.imread(pca1_path)
            pca2_image = cv2.imread(pca2_path)
            images = np.concatenate((rgb_image, pca1_image, pca2_image), axis=2)
            img = self.transform(images)
            return img, rgb_path
        else:
            rgb_path = self.images_path[index]
            pca1_path = rgb_path.replace("RGB", "PCA1")
            pca2_path = rgb_path.replace("RGB", "PCA2")

            str_1 = rgb_path.split("\\")
            str_2 = str_1[len(str_1) - 1].split("-")
            label = int(str_2[1])  # 获取簇的标签

            rgb_image = cv2.imread(rgb_path)
            pca1_image = cv2.imread(pca1_path)
            pca2_image = cv2.imread(pca2_path)
            images = np.concatenate((rgb_image, pca1_image, pca2_image), axis=2)
            img = self.transform(images)
            # 随机进行数据增强，为2时不处理
            # flipCote = random.choice([-1, 0, 1, 2])
            # if flipCote != 2:
            #   image = self.augment(image, flipCote)
            return img, label - 1

    # 返回训练集大小
    def __len__(self):
        return len(self.images_path)
