import torch
import time
from torch import nn
from tqdm import tqdm
from resnet import base_resnet
import os


# 训练
def train_valid(net_type, train_loader, val_loader, learning_rate, epochs, factor, data_path):
    net_model = base_resnet(net_type)
    print(net_model)
    if torch.cuda.is_available():  # GPU是否可用
        net_model = net_model.cuda()
    # 定义损失
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(net_model.parameters(), learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210, 240], gamma=0.50)
    accuracy = 0.0  # 保存精度
    cur_train_time = 0.0  # 统计当前分裂的训练时间
    # 保存训练模型
    model_save_path = data_path + '\\model'  # 初始化训练集路径
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    # 训练
    start = time.time()  # 计算训练的时间
    for epoch in range(epochs + 1):
        i = 0
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch), ncols=100) as tq:
            for image, label in tqdm(train_loader):
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()
                out = net_model(image)
                predict_label = out[0]
                loss_1 = criterion(predict_label, label.long())
                loss_2 = torch.norm(torch.norm(out[1], p=1, dim=0), p=2).pow(2)  # L12-norm
                loss = loss_1 + factor * loss_2
                i += 1
                tq.set_postfix({'lr': '%.5f' % optimizer.param_groups[0]['lr'], 'loss': '%.4f' % (loss.item())})
                tq.update(1)
                torch.cuda.empty_cache()
                # 反向传播，更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 交叉熵验证
        if epoch >= 180 and epoch % 30 == 0:
            acc = start_valid(net_model, val_loader)
            print(acc[0], acc[1], acc[2])
            net_model.train()
            if str(acc[2]) == 'nan':
                print('验证精度结果异常，继续训练')
                continue
            elif acc[2] >= accuracy:
                accuracy = acc[2]
                torch.save(net_model.state_dict(), '%s\\down_best_model.pth' % model_save_path)
        scheduler.step()
    if accuracy == 0:
        print('模型训练失败，程序退出')
        exit(0)
    else:
        print('训练完毕，返回模型')
        net_model.load_state_dict(torch.load('%s\\down_best_model.pth' % model_save_path))
        acc = start_valid(net_model, val_loader)
        print(acc[0], acc[1], acc[2])
        end = time.time()
        cur_train_time = cur_train_time + end - start

    return net_model, cur_train_time


# 验证
def start_valid(net_model, val_loader):
    # 精度指标
    TP = 0  # 被模型预测为正的正样本 (0:正列,1:反例)
    FP = 0  # 被模型预测为正的负样本
    FN = 0  # 被模型预测为负的正样本
    TN = 0  # 被模型预测为负的负样本
    # 精度指标
    net_model.eval()  # 验证模式
    for image, label in tqdm(val_loader):
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        torch.cuda.empty_cache()
        output = net_model(image)
        predict_label = output[0]
        predict_label = predict_label.detach().max(1)[1]
        for i in range(label.size(0)):
            if predict_label[i] == 0 and label[i] == 0:
                TP += 1
                continue
            if predict_label[i] == 0 and label[i] == 1:
                FP += 1
                continue
            if predict_label[i] == 1 and label[i] == 0:
                FN += 1
                continue
            if predict_label[i] == 1 and label[i] == 1:
                TN += 1
                continue
    eval_oa = 100 * (TP + TN) / (TP + FN + FP + TN)
    eval_acc = 100 * TP / (TP + FP + 1)
    eval_call = 100 * TP / (TP + FN + 1)
    eval_f1 = 2 * (eval_acc * eval_call) / (eval_acc + eval_call + 1)
    return eval_oa, eval_acc, eval_f1
