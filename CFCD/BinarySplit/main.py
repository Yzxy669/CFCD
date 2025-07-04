import torchvision
from Tool import DataProduction as dp
from label_redefinition import refine_label_tv
from label_redefinition import training_sample_cluster
from predict import *
from train_and_verif import train_valid
from convert_image import to_image

if __name__ == '__main__':
    # 参数设置
    path_gt = "Data path"  # 数据对应的真值文件
    path = "Save path"  # 数据保存的主干路径
    samples_num = samples_num  # 未知类别的训练样本数据的个数
    factor = 1*1e-4  # 可根据具体数据调节大小
    batch_size = batch_size 
    learning_rate = 1 * 1e-2
    epochs = 500  # 最大训练次数
    net_type = 'resnet18'  # 网络类型
    # 参数设置
    splits_num = 0  # 初始化分裂次数
    train_time = 0  # 计算训练的总时间
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    while True:
        path_main = path + "\\Split-%d" % splits_num  # 更新主路径
        if not os.path.exists(path_main):  # 如果没有继续分裂的文件，程序自动结束
            print("没有继续可分裂的文件，正在进行图像转换")
            print("训练花费的时间%f" % train_time)
            path_main = path
            to_image(path_gt, path_main)
            print("图像转换完成，算法结束")
            exit(0)
        train_name = os.listdir(path_main + "\\Train")  # 获取训练集的所有文件名
        valid_name = os.listdir(path_main + "\\ValidSet")  # 获取验证集的所有文件名
        test_name = os.listdir(path_main + "\\Test")  # 获取测试集的所有文件名
        id_run_num = 0  # 获取有效执行次数
        for run_num in range(len(train_name)):
            path_train = path_main + "\\Train\\" + train_name[run_num]  # 获取当前训练样本路径
            path_valid = path_main + "\\ValidSet\\" + valid_name[run_num]  # 获取当前验证集路径
            path_test = path_main + "\\Test\\" + test_name[run_num]  # 获取当前测试集路径
            id_run_num = int((train_name[run_num].split('-'))[1]) - 1  # 获取可分裂的无标签样本编号
            # 为训练样本聚类，重写验证集和训练集标签
            cluster_id, k_means_labels = training_sample_cluster(path_train, samples_num, splits_num)
            num_split_class = 0  # num_split_class = 2 时说明可以分裂,如果num_split_class = 1说明不可分裂
            for i in range(len(cluster_id)):
                if len(cluster_id[i]) != 0:
                    num_split_class = num_split_class + 1
            if num_split_class == 2:
                print("##########################第%d次分裂######################" % (splits_num + 1))
                # 说明当前训练样本可以聚为2类
                print('重写训练集和验证集聚类簇标签')
                refine_label_tv(cluster_id, path_main, splits_num, id_run_num)  # 重写训练集和验证集标签
                # 加载训练数据
                is_bi_dataset = dp.ISBI_Loader(path_train, transform=torchvision.transforms.ToTensor())
                train_loader = torch.utils.data.DataLoader(dataset=is_bi_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True)
                print("训练数据个数：", len(is_bi_dataset))

                is_bi_dataset = dp.ISBI_Loader(path_valid, transform=torchvision.transforms.ToTensor())
                val_loader = torch.utils.data.DataLoader(dataset=is_bi_dataset,
                                                         batch_size=256,
                                                         shuffle=False)
                print("验证集数据个数：", len(is_bi_dataset))
                # 训练模型
                print("开始训练.....")
                net_model, cur_train_time = train_valid(net_type, train_loader, val_loader, learning_rate, epochs,
                                                        factor, path_main)
                train_time = train_time + cur_train_time  # 统计分裂训练的总时间
                # 测试数据
                is_bi_dataset = dp.ISBI_Loader(path_test, transform=torchvision.transforms.ToTensor(),
                                               data_type='Test')
                test_loader = torch.utils.data.DataLoader(dataset=is_bi_dataset,
                                                          batch_size=256,
                                                          shuffle=False)
                print("测试集个数：", len(is_bi_dataset))
                # 预测数据标签，将预测后数据按照类别写入对应的文件
                print("开始测试.....")
                predict(net_model, test_loader, path_main, splits_num, id_run_num)
            else:
                # 说明当前训练样本已不能再分裂 ，将对应的当前分裂好的测试数据写入最终的测试数据集
                print("正在写已经分裂完成的测试数据.....")
                c_label = cluster_id[0] if cluster_id[0] else cluster_id[1]  # 若当前训练集不可再分，那么就将测试集标签和此时的样本标签写为一致
                path_test_final = path + "\\Test_Final\\"
                if not os.path.exists(path_test_final):
                    os.makedirs(path_test_final)
                path_test_over = glob.glob(os.path.join('%s\\RGB\\*.png' % path_test))
                for i in range(len(path_test_over)):
                    image = cv2.imread(path_test_over[i])
                    str_1 = path_test_over[i].split("\\")
                    str_2 = str_1[len(str_1) - 1].split("-")
                    path_save_test = path_test_final + "%d-%s-%s" % (
                        c_label[0], str_2[len(str_2) - 2], str_2[len(str_2) - 1])
                    cv2.imwrite(path_save_test, image)
                    os.remove(path_test_over[i])  # 为节省内存空间，清除已经分裂完毕的测试数据，如果想保存每一次的分裂文件，注释下面三行代码
                    os.remove(path_test_over[i].replace("RGB", "PCA1"))
                    os.remove(path_test_over[i].replace("RGB", "PCA2"))
        splits_num = splits_num + 1  # 分裂次数自增1
