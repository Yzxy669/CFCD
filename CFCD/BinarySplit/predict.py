import torch
import glob
import cv2
import numpy as np
import os
from torchvision import transforms
from features_to_text import *
from tqdm import tqdm


# 测试模式,预测整张影像的标签
def predict(net_model, test_loader, path_main, splits_num, run_num):
    net_model.eval()  # 模型处于测试类型
    # 训练样本提取通道注意力特征
    torch.cuda.empty_cache()
    path_main = path_main.replace("Split-%d" % splits_num, "Split-%d" % (splits_num + 1))
    sample_paths = [path_main + "\\Train\\Train-%d\\RGB" % (2 * run_num + 1),
                    path_main + "\\Train\\Train-%d\\RGB" % (2 * run_num + 2)]
    for path_i in range(len(sample_paths)):
        features_train = []  # 存储训练数据的深度特征
        train_sample_path = glob.glob(os.path.join('%s\\*.png' % sample_paths[path_i]))
        for j in range(len(train_sample_path)):
            RGB_image = cv2.imread(train_sample_path[j])
            PCA1_image = cv2.imread(train_sample_path[j].replace("RGB", "PCA1"))
            PCA2_image = cv2.imread(train_sample_path[j].replace("RGB", "PCA2"))
            images = np.concatenate((RGB_image, PCA1_image, PCA2_image), axis=2)
            tensor_trans = transforms.ToTensor()
            image = tensor_trans(images)  # 将Images影像的维度数轴变换
            image = torch.unsqueeze(image, 0)  # 将三维的Tensor包装成四维
            output = net_model(image.cuda())
            feats = output[1].cuda().data.cpu().tolist()  # 提取到的深度特征
            features_train.append(feats)
        feature_path = sample_paths[path_i].split("RGB")
        feature_text(features_train, feature_path[0])

    # 测试数据分类
    for images, path_image in tqdm(test_loader):
        torch.cuda.empty_cache()
        output = net_model(images.cuda())
        predict_label = output[0]
        predict_label = predict_label.cuda().data.cpu().numpy()
        label = np.argmax(predict_label, axis=1)  # 获取每行数据中最大值的下标
        for i in range(len(label)):
            rgb_image = cv2.imread(path_image[i])  # 读取RGB图像
            pca1_image = cv2.imread(path_image[i].replace("RGB", "PCA1"))  # 读取PCA1图像
            pca2_image = cv2.imread(path_image[i].replace("RGB", "PCA2"))  # 读取PCA2图像
            str_1 = path_image[i].split("\\")
            str_2 = str_1[len(str_1) - 1].split("-")
            if label[i] + 1 == 1:
                rgb_path = path_main + "\\Test\\Test-%d\\RGB\\1-%s-%s" % (
                    2 * run_num + 1, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(rgb_path, rgb_image)
                pca1_path = path_main + "\\Test\\Test-%d\\PCA1\\1-%s-%s" % (
                    2 * run_num + 1, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(pca1_path, pca1_image)
                pca2_path = path_main + "\\Test\\Test-%d\\PCA2\\1-%s-%s" % (
                    2 * run_num + 1, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(pca2_path, pca2_image)
            elif label[i] + 1 == 2:
                rgb_path = path_main + "\\Test\\Test-%d\\RGB\\2-%s-%s" % (
                    2 * run_num + 2, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(rgb_path, rgb_image)
                pca1_path = path_main + "\\Test\\Test-%d\\PCA1\\2-%s-%s" % (
                    2 * run_num + 2, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(pca1_path, pca1_image)
                pca2_path = path_main + "\\Test\\Test-%d\\PCA2\\2-%s-%s" % (
                    2 * run_num + 2, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(pca2_path, pca2_image)
                #  为节省内存空间，清除已经使用过的测试数据，如果想保存每一次的分裂文件，注释下面三行代码
            os.remove(path_image[i])
            os.remove(path_image[i].replace("RGB", "PCA1"))
            os.remove(path_image[i].replace("RGB", "PCA2"))
