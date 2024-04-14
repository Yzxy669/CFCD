from sklearn.cluster import KMeans
import glob
import os
import cv2
import numpy as np


# 以裂变二分类的思想返回训练样本数据的标签
def training_sample_cluster(train_path, samples_num, splits_num):
    feature_list = []  # 存储第一次聚类的特征
    if splits_num == 0:
        rgb_path = glob.glob(os.path.join('%s\\RGB\\*.png' % train_path))
        pca1_path = glob.glob(os.path.join('%s\\PCA1\\*.png' % train_path))
        pca2_path = glob.glob(os.path.join('%s\\PCA2\\*.png' % train_path))
        for i in range(len(rgb_path)):
            rgb_image = cv2.imread(rgb_path[i])
            pca1_image = cv2.imread(pca1_path[i])
            pca2_image = cv2.imread(pca2_path[i])
            images = np.concatenate(([rgb_image, pca1_image, pca2_image]), axis=2)
            feature = list(np.array(images).flatten())
            feature_list.append(feature)
        k_means = KMeans(n_clusters=2, init='k-means++', random_state=0)
        k_means.fit(feature_list)
        k_means_labels = k_means.predict(feature_list)
        cluster_id = label_set(k_means_labels, samples_num, train_path)
    else:
        path_feature = train_path + "\\feature.txt"  # 获取存放深度特征的文件路径
        all_features = []  # 存放txt中断所有特征
        if os.path.exists(path_feature):
            f = open(path_feature)
            while True:
                lines = f.readline()
                if len(lines) == 0:
                    f.close()
                    break
                str_1 = lines.split(',')
                features = [float(i) for i in str_1]
                all_features.append(features)
        else:
            print("找不到对应的特征文件,算法退出")  # 如果找不到对应的特征文件，程序中断
            exit(0)
            # 使用深度特征聚类
        k_means = KMeans(n_clusters=2, init='k-means++', random_state=0)
        k_means.fit(all_features)
        k_means_labels = k_means.predict(all_features)
        cluster_id = label_set(k_means_labels, samples_num, train_path)
    return cluster_id, k_means_labels


# 返回簇的标签
def label_set(k_means_labels, sample_num, train_path):
    label_list = [[], []]  # 创建簇标签列表
    image_path = glob.glob(os.path.join('%s\\RGB\\*.png' % train_path))
    if len(k_means_labels) <= 2.0 * sample_num:  # 只有两类，那就各分为一类；或只有一类，直接划分为第1类
        for i in range(int(len(k_means_labels) / sample_num)):
            str_1 = image_path[(i + 1) * sample_num - 1].split("\\")
            str_2 = str_1[len(str_1) - 1].split("-")
            label_list[i].append(int(str_2[0]))
    else:  # 大于两类，必须划分为两类
        not_two = True  # 是否为两类 ，初始不是两类，为True
        rate = -0.025  # 簇标签偏向权值初始设置为-0.05
        sum_label = 0.0  # 统计聚类标签的和
        for i in range(len(k_means_labels)):
            sum_label = sum_label + k_means_labels[i] + 1
        while not_two:
            label_list[0].clear()  # 清空簇标签列表0
            label_list[1].clear()  # 清空簇标签列表1
            rate = rate + 0.025  # 权值动态递增，直到可以分为2簇为止
            print('无标签样本正在划分簇中，当前rate = %f' % rate)
            if rate >= 0.5:
                print('当前未知样本无法划分，程序退出')
                exit(0)
            for i in range(int(len(k_means_labels) / sample_num)):
                str_1 = image_path[(i + 1) * sample_num - 1].split("\\")
                str_2 = str_1[len(str_1) - 1].split("-")
                count = 0.0  # 计算每簇样本的标签和
                for j in range(i * sample_num, (i + 1) * sample_num):
                    count = count + k_means_labels[j] + 1
                if sum_label <= len(k_means_labels) * 1.5:  # 说明聚类编号为0的过多,偏向第1类
                    if count < sample_num * (1.5 - rate):
                        label_list[0].append(int(str_2[0]))
                    else:
                        label_list[1].append(int(str_2[0]))
                else:  # 说明聚类编号为1的过多，偏向第0类
                    if count < sample_num * (1.5 + rate):
                        label_list[0].append(int(str_2[0]))
                    else:
                        label_list[1].append(int(str_2[0]))
            if len(label_list[0]) != 0 and len(label_list[1]) != 0:
                if int(len(k_means_labels) / sample_num) > 5:
                    if rate >= 0.40:  # 说明当前已经无法调整
                        not_two = False
                        print('无标签样本划分完成')
                    elif abs(len(label_list[0]) - len(label_list[1])) < (
                            len(k_means_labels) / sample_num) - 2:  # 为避免样本失衡
                        not_two = False  # 说明当前已经分为两类
                        print('无标签样本划分完成')
                else:
                    not_two = False  # 说明当前未知样本类别过少,当前已经无法调整
                    print('无标签样本划分完成')

    return label_list


# 重新写标签为训练数据和验证集
def refine_label_tv(cluster_id, path_main, splits_num, run_num):
    # 创建为下一次分裂存放数据的文件夹
    path_new_splits = path_main.replace("Split-%d" % splits_num, "Split-%d" % (splits_num + 1))
    file_path = [path_new_splits + "\\Train\\Train-%d\\RGB" % (2 * run_num + 1),
                 path_new_splits + "\\Train\\Train-%d\\PCA1" % (2 * run_num + 1),
                 path_new_splits + "\\Train\\Train-%d\\PCA2" % (2 * run_num + 1),
                 path_new_splits + "\\Train\\Train-%d\\RGB" % (2 * run_num + 2),
                 path_new_splits + "\\Train\\Train-%d\\PCA1" % (2 * run_num + 2),
                 path_new_splits + "\\Train\\Train-%d\\PCA2" % (2 * run_num + 2),
                 path_new_splits + "\\ValidSet\\ValidSet-%d\\RGB" % (2 * run_num + 1),
                 path_new_splits + "\\ValidSet\\ValidSet-%d\\PCA1" % (2 * run_num + 1),
                 path_new_splits + "\\ValidSet\\ValidSet-%d\\PCA2" % (2 * run_num + 1),
                 path_new_splits + "\\ValidSet\\ValidSet-%d\\RGB" % (2 * run_num + 2),
                 path_new_splits + "\\ValidSet\\ValidSet-%d\\PCA1" % (2 * run_num + 2),
                 path_new_splits + "\\ValidSet\\ValidSet-%d\\PCA2" % (2 * run_num + 2),
                 path_new_splits + "\\Test\\Test-%d\\RGB" % (2 * run_num + 1),
                 path_new_splits + "\\Test\\Test-%d\\PCA1" % (2 * run_num + 1),
                 path_new_splits + "\\Test\\Test-%d\\PCA2" % (2 * run_num + 1),
                 path_new_splits + "\\Test\\Test-%d\\RGB" % (2 * run_num + 2),
                 path_new_splits + "\\Test\\Test-%d\\PCA1" % (2 * run_num + 2),
                 path_new_splits + "\\Test\\Test-%d\\PCA2" % (2 * run_num + 2)]
    for i in range(len(file_path)):
        if not os.path.exists(file_path[i]):
            os.makedirs(file_path[i])

    # 为训练集,验证集重写标签
    train_path = path_main + "\\Train\\Train-%d" % (run_num + 1)  # 前一次的训练路径
    type_data = ["RGB", "PCA1", "PCA2"]
    for i in range(len(type_data)):
        image_path = glob.glob(os.path.join('%s\\%s\\*.png' % (train_path, type_data[i])))
        for j in range(len(image_path)):
            image = cv2.imread(image_path[j], flags=-1)
            o_label = int(image[0, 0, 3])  # 图像最初的原始标签
            str_1 = image_path[j].split("\\")
            str_2 = str_1[len(str_1) - 1].split("-")
            if o_label in cluster_id[0]:
                save_path = file_path[i] + "\\%d-1-%s-%s" % (
                    o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(save_path, image)
                save_path = train_path + "\\%s\\%d-1-%s-%s" % (
                    type_data[i], o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                os.remove(image_path[j])
                cv2.imwrite(save_path, image)
            elif o_label in cluster_id[1]:
                save_path = file_path[i + 3] + "\\%d-2-%s-%s" % (
                    o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(save_path, image)
                save_path = train_path + "\\%s\\%d-2-%s-%s" % (
                    type_data[i], o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                os.remove(image_path[j])
                cv2.imwrite(save_path, image)

        valid_set_path = path_main + "\\ValidSet\\ValidSet-%d" % (run_num + 1)  # 前一次的验证集路径
        image_path = glob.glob(os.path.join('%s\\%s\\*.png' % (valid_set_path, type_data[i])))
        for k in range(len(image_path)):
            image = cv2.imread(image_path[k], flags=-1)
            o_label = int(image[0, 0, 3])  # 图像最初的原始标签
            str_1 = image_path[k].split("\\")
            str_2 = str_1[len(str_1) - 1].split("-")
            if o_label in cluster_id[0]:
                save_path = file_path[i + 6] + "\\%d-1-%s-%s" % (
                    o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(save_path, image)

                save_path = valid_set_path + "\\%s\\%d-1-%s-%s" % (
                    type_data[i], o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                os.remove(image_path[k])
                cv2.imwrite(save_path, image)
            elif o_label in cluster_id[1]:
                save_path = file_path[i + 9] + "\\%d-2-%s-%s" % (
                    o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                cv2.imwrite(save_path, image)
                save_path = valid_set_path + "\\%s\\%d-2-%s-%s" % (
                    type_data[i], o_label, str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                os.remove(image_path[k])
                cv2.imwrite(save_path, image)
