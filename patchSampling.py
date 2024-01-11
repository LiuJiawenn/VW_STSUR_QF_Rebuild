import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import matplotlib.pyplot as plt


def extract_patches(arr, patch_shape=(32, 32, 3), extraction_step=(32, 32, 3)):
    # input (1080,1920,3) output(1980,3,32,32)
    # 对应维度+1需要跳过的字节数
    patch_strides = arr.strides
    # 三个维度，每个维度创建一个切片
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    # patch_indices_shape = (33,60,1) 即每个维度能切出来的patch数
    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    patches = np.transpose(patches.reshape((-1, patch_shape[0], patch_shape[1], patch_shape[2])), (0, 3, 1, 2))
    return patches


def source_optical_flow_patch_extract(video_path, keyframes):
    # 该函数在视频上找到关键帧，整理出关键帧与光流的patch列表
    # 打开AVI视频
    cap = cv2.VideoCapture(video_path)
    s_patches = []
    o_patches = []

    # 依次定位到关键帧
    for k in keyframes:
        if k == 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k-2)

        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 分别计算这一帧以及光流的所有patch。用list收集起来
        s_patch = extract_patches(frame2, (32, 32, 3), (32, 32, 3))
        o_patch = extract_patches(flow.reshape(1080, 1920, 2), (32, 32, 2), (32, 32, 2))
        s_patches.append(s_patch)
        o_patches.append(o_patch)

    cap.release()
    return s_patches, o_patches


def key_patch_select(s_patches, c_patches, patch_per_frame=64, patch_rate=0.4):
    # 该函数按照psnr 输出关键patch的索引
    # 通常情况下 s_patches，c_patches应该是 12帧×1980样本数×3×32×32 的patch列表
    # 选关键patch
    frame_nb = len(s_patches)
    patch_nb = len(s_patches[0])
    # psnr_values = np.zeros(num_patches)
    key_patch_list = []
    ratio = int(patch_rate * patch_nb)

    # 按帧选择样本，计算每个patch的psnr，选择范围在%r小的（r = 0.4） 小的psnr patch上 随机选64个
    for i in range(frame_nb):
        psnr_temp = [] # 存储一帧上所有样本的psnr
        for j in range(patch_nb):
            psnr_temp.append(psnr(s_patches[i][j], c_patches[i][j]))

        sorted_indices = np.argsort(psnr_temp)  # 按psnr排序，输出是排序后的索引
        index_mask = np.random.choice(ratio, patch_per_frame, replace=False)  # 随机选64个patch
        sorted_indices = sorted_indices[index_mask]  # 找到随机patch的索引
        sorted_indices.sort()  # 排下序
        key_patch_list.append(sorted_indices)  # 取出对应的patch

    return key_patch_list


def key_patch_mask(s_patches,c_patches,so_patches,co_patches,key_patch_index):
    # 该函数关键patch的索引 取出关键patch整理成列表
    key_s_patches = []
    key_c_patches = []
    key_so_patches = []
    key_co_patches = []

    for i in range(len(key_patch_index)):
        key_list = key_patch_index[i]
        key_s_patches.append(s_patches[i][key_list])
        key_c_patches.append(c_patches[i][key_list])
        key_so_patches.append(so_patches[i][key_list])
        key_co_patches.append(co_patches[i][key_list])

    # 先用list 收集,最后连起来变成numpy矩阵,比较高效
    key_s_patches = np.concatenate(key_s_patches)
    key_c_patches = np.concatenate(key_c_patches)
    key_so_patches = np.concatenate(key_so_patches)
    key_co_patches = np.concatenate(key_co_patches)

    # plt.imshow(np.transpose(key_s_patches[12], (1, 2, 0)))
    # plt.imshow(np.transpose(key_c_patches[12], (1, 2, 0)))
    # plt.imshow(key_so_patches[12][0])
    # plt.imshow(key_co_patches[12][0])

    return torch.from_numpy(key_s_patches), torch.from_numpy(key_c_patches), torch.from_numpy(key_so_patches), torch.from_numpy(key_co_patches)
