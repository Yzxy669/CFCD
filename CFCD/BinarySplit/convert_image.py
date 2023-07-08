import cv2
import numpy as np
import os
import glob


def to_image(path_gt, path_main):
    gt_image = cv2.imread(path_gt)
    image_size = gt_image.shape
    target_img = np.zeros([image_size[0], image_size[1], 3], dtype=np.uint8)
    train_image_path = glob.glob(os.path.join('%s\\Split-0\\Train\\Train-1\\RGB\\*.png' % path_main))
    test_image_path = glob.glob(os.path.join('%s\\Test_Final\\*.png' % path_main))
    save_path = path_main + "\\Test_Final"
    for i in range(len(train_image_path)):
        train_path = train_image_path[i]
        str_1 = train_path.split('\\')
        str_2 = str_1[len(str_1) - 1].split('-')
        str_3 = str_2[len(str_2)-1].split('.')
        label = int(str_2[0])
        x = int(str_2[len(str_2) - 2])
        y = int(str_3[0])
        target_img[x, y, :] = label  # B通道赋值
    for j in range(len(test_image_path)):
        test_path = test_image_path[j]
        str_1 = test_path.split('\\')
        str_2 = str_1[len(str_1) - 1].split('-')
        str_3 = str_2[len(str_2) - 1].split('.')
        label = int(str_2[0])
        x = int(str_2[len(str_2) - 2])
        y = int(str_3[0])
        target_img[x, y, :] = label  # 三通道赋值
    cv2.imwrite(save_path + "\\test_image.bmp", target_img)
