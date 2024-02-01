import numpy as np
import os
import ffmpeg
from patchSampling import source_optical_flow_patch_extract, key_patch_select, key_patch_mask


def h264_to_avi(path_264, path_avi):
    # 解压视频用
    # 我的C盘读写加上解压视频的时间都比磁盘直接读写快
    # 学姐要是有磁盘存并且磁盘读写速度足够快,可以解压完再训练
    if os.path.exists(path_avi):
        return

    stream = ffmpeg.input(path_264)
    stream = ffmpeg.output(stream, path_avi, vcodec='rawvideo')
    ffmpeg.run(stream)


def data_generator(videoList = list(range(44, 220)), shuffle = True, patch_per_frame=64, patch_rate=0.4):
    # TODO:
    # 1. 数据生成器,每次在 yield关键字处返回每次需要输入网络的patch tensor

    # 存储.264文件 夹 路径
    raw_path = 'D:/rawVideo264/'
    # 解压视频用的C盘文件夹
    running_folder = 'C:/Users/Hong_Lab/Desktop/runningFolder/'  # C盘读写快
    # 提前计算好的关键帧列表
    kf12_list = np.load('data/kf12_list.npy')
    # SUR
    SUR_GROUND_TRUTH = np.load("data/SUR_GROUND_TRUTH.npy")
    # trainVideos = np.load("data/trainsetVideos.npy")
    trainVideos = videoList  # 在list里写上训练视频编号即可范围0-219
    if shuffle:
        np.random.shuffle(trainVideos)  # 打乱训练顺序

    dir_list = os.listdir(raw_path)  # 可以返回220个文件夹名,帮我索引
    # 45-176是训练集,就是video index
    # for i in range(44, 176):
    # 下面开始随机的训练集，并且每个epoch打乱
    for i in trainVideos:
        current_video = dir_list[i]
        s_264 = raw_path + current_video + '/' + current_video + '_qp_00.264'
        s_avi = running_folder + current_video + '_qp_00.avi'
        h264_to_avi(s_264, s_avi)
        s_patches, so_patches = source_optical_flow_patch_extract(s_avi, kf12_list[i])

        # for qp in range(1, 52):
        # for qp in [1, 8, 15, 22, 29, 36, 43, 50]:
        for qp in [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 51]:
            c_264 = raw_path + current_video + '/' + current_video + '_qp_'+'{:02d}'.format(qp)+'.264'
            c_avi = running_folder + current_video + '_qp_'+'{:02d}'.format(qp)+'.avi'
            h264_to_avi(c_264, c_avi)
            c_patches, co_patches = source_optical_flow_patch_extract(c_avi, kf12_list[i])
            key_patch_index = key_patch_select(s_patches, c_patches, patch_per_frame=patch_per_frame, patch_rate=patch_rate)
            key_s_patches, key_c_patches, key_so_patches, key_co_patches = key_patch_mask(s_patches, c_patches, so_patches, co_patches, key_patch_index)

            yield (key_s_patches, key_c_patches, key_so_patches, key_co_patches), SUR_GROUND_TRUTH[i][qp-1]
            # 删除上一个压缩视频
            os.remove(c_avi)

        # 删除上一个原视频
        os.remove(s_avi)