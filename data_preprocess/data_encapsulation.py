"""
本py程序为数据封装程序 负责如下功能:
    1. 从csv文件读入数据 (label.csv & 递归去读入其他的csv)
    2. 进行标准化处理
    3. Temporal数据 降采样 二维化处理
    4. Spectral数据 进行二维化处理
    5. shuttle
    6. 打包数据集为npy: 标签数据 时域数据 空域数据
    0. 数据集地址:
        rain_specInput_root_path = ../database/train_spectral
        train_tempInput_root_path = ../database/train_temporal
        train_label_root_path = ../database/train_label

        test_specInput_root_path = ../database/test_spectral
        test_tempInput_root_path = ../database/test_temporal
        test_label_root_path = ../database/test_label
"""

import os
import numpy as np

# 数据根目录
data_root_path_temporal = '../data_temporal'
data_root_path_spectral = '../data_spectral'

# 数据存储目录
data_root_path = '../database'

# 标签数据读入转存到np
label = np.array(np.loadtxt(os.path.join(
    data_root_path_temporal, f"Tag.csv"), dtype=float, delimiter=',', skiprows=1))
print("label shape")
print(np.shape(label))

# 时域数据读入 (60, 9, 9,32768)
# 剔除21 18 两组未处理数据 剔除20一组 采样不够数据 同时规范行数为32768
temporal_data_sets = []
for i in range(21):
    for j in range(3):
        if (i + 1 == 20 and j + 1 == 2) or (i + 1 == 21 and j + 1 == 2) or (i + 1 == 18 and j + 1 == 2):
            ...
        else:
            data_set = np.array(np.loadtxt(os.path.join(
                data_root_path_temporal, f"{i + 1}/{j + 1}/eeg.csv"), dtype=float, delimiter=',', skiprows=1,
                usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))

            # # 测试不合格数据
            # if data_set.shape[0] < 32768:
            #     print("i,j,datashape0")
            #     print(i + 1)
            #     print(j + 1)
            #     print(data_set.shape[0])

            # standardization
            data_set = (data_set - data_set.mean()) / data_set.std()
            data_set = data_set[0:32768, :]

            # 数据格式映射
            # EEG.AF3	EEG.F7	EEG.F3	EEG.FC5	EEG.T7	EEG.P7	EEG.O1	EEG.O2	EEG.P8	EEG.T8	EEG.FC6	EEG.F4	EEG.F8	EEG.AF4
            # (1,3)     (2,0)   (2,2)   (3,1)   (4,0)   (6,0)   (8,3)   (8,5)   (6,8)   (4,8)   (3,7)   (2,6)   (2,8)   (1,5)
            empty_set = np.zeros(shape=(32768, 9, 9), dtype=float)
            for m in range(empty_set.shape[0]):
                empty_img = np.zeros(shape=(9, 9), dtype=float)
                for n in range(14):
                    if n == 0:
                        empty_img[1][3] = data_set[m][n]
                    if n == 1:
                        empty_img[2][0] = data_set[m][n]
                    if n == 2:
                        empty_img[2][2] = data_set[m][n]
                    if n == 3:
                        empty_img[3][1] = data_set[m][n]
                    if n == 4:
                        empty_img[4][0] = data_set[m][n]
                    if n == 5:
                        empty_img[6][0] = data_set[m][n]
                    if n == 6:
                        empty_img[8][3] = data_set[m][n]
                    if n == 7:
                        empty_img[8][5] = data_set[m][n]
                    if n == 8:
                        empty_img[6][8] = data_set[m][n]
                    if n == 9:
                        empty_img[4][8] = data_set[m][n]
                    if n == 10:
                        empty_img[3][7] = data_set[m][n]
                    if n == 11:
                        empty_img[2][6] = data_set[m][n]
                    if n == 12:
                        empty_img[2][8] = data_set[m][n]
                    if n == 13:
                        empty_img[1][5] = data_set[m][n]
                empty_set[m] = empty_img
            data_set = empty_set
            temporal_data_sets.append(data_set)
temporal_data_sets = np.array(temporal_data_sets)
temporal_data_sets = np.transpose(temporal_data_sets, [0, 2, 3, 1])
print("temporal shape")
print(np.shape(temporal_data_sets))

# 频域数据读入 (60, 9, 9, 5)
spectral_data_sets = []
for i in range(21):
    for j in range(3):
        if (i + 1 == 20 and j + 1 == 2) or (i + 1 == 21 and j + 1 == 2) or (i + 1 == 18 and j + 1 == 2):
            ...
        else:
            data_set = np.array(np.loadtxt(os.path.join(
                data_root_path_spectral, f"{i + 1}/{i + 1}_{j + 1}_DE.csv"), dtype=float, delimiter=','))

            # standardization
            data_set = (data_set - data_set.mean()) / data_set.std()
            # 数据格式映射
            # EEG.AF3	EEG.F7	EEG.F3	EEG.FC5	EEG.T7	EEG.P7	EEG.O1	EEG.O2	EEG.P8	EEG.T8	EEG.FC6	EEG.F4	EEG.F8	EEG.AF4
            # (1,3)     (2,0)   (2,2)   (3,1)   (4,0)   (6,0)   (8,3)   (8,5)   (6,8)   (4,8)   (3,7)   (2,6)   (2,8)   (1,5)
            empty_set = np.zeros(shape=(5, 9, 9), dtype=float)
            for m in range(empty_set.shape[0]):
                empty_img = np.zeros(shape=(9, 9), dtype=float)
                for n in range(14):
                    if n == 0:
                        empty_img[1][3] = data_set[m][n]
                    if n == 1:
                        empty_img[2][0] = data_set[m][n]
                    if n == 2:
                        empty_img[2][2] = data_set[m][n]
                    if n == 3:
                        empty_img[3][1] = data_set[m][n]
                    if n == 4:
                        empty_img[4][0] = data_set[m][n]
                    if n == 5:
                        empty_img[6][0] = data_set[m][n]
                    if n == 6:
                        empty_img[8][3] = data_set[m][n]
                    if n == 7:
                        empty_img[8][5] = data_set[m][n]
                    if n == 8:
                        empty_img[6][8] = data_set[m][n]
                    if n == 9:
                        empty_img[4][8] = data_set[m][n]
                    if n == 10:
                        empty_img[3][7] = data_set[m][n]
                    if n == 11:
                        empty_img[2][6] = data_set[m][n]
                    if n == 12:
                        empty_img[2][8] = data_set[m][n]
                    if n == 13:
                        empty_img[1][5] = data_set[m][n]
                empty_set[m] = empty_img
            data_set = empty_set
            spectral_data_sets.append(data_set)
spectral_data_sets = np.array(spectral_data_sets)
spectral_data_sets = np.transpose(spectral_data_sets, [0, 2, 3, 1])
print("spectral shape")
print(np.shape(spectral_data_sets))

# shuffle
index = np.arange(label.shape[0])
np.random.shuffle(index)

temporal_data_sets = temporal_data_sets[index]
spectral_data_sets = spectral_data_sets[index]
label = label[index]

# 划分训练集 测试集 (训练集 测试集 验证集 = 6:2:2)
train_temporal = temporal_data_sets[0:48, :, :, :]
train_spectral = spectral_data_sets[0:48, :, :, :]
train_label = label[0:48, ]
print("train_spectral")
print(np.shape(train_spectral))

test_temporal = temporal_data_sets[48:, :, :, :]
test_spectral = spectral_data_sets[48:, :, :, :]
test_label = label[48:, ]
print("text_spectral")
print(np.shape(test_spectral))

# 数据打包
if not os.path.exists(data_root_path):
    os.mkdir(data_root_path)
np.save(os.path.join(data_root_path, f"train_temporal.npy"), train_temporal)
np.save(os.path.join(data_root_path, f"train_spectral.npy"), train_spectral)
np.save(os.path.join(data_root_path, f"train_label.npy"), train_label)
np.save(os.path.join(data_root_path, f"test_temporal.npy"), test_temporal)
np.save(os.path.join(data_root_path, f"test_spectral.npy"), test_spectral)
np.save(os.path.join(data_root_path, f"test_label.npy"), test_label)
