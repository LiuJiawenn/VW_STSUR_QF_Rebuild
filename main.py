import os
import ffmpeg
import numpy as np
from scipy.stats import truncnorm
import torch.nn.init as init
from patchSampling import source_optical_flow_patch_extract, key_patch_select, key_patch_mask
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import traceback
from model.STSUR_Net import STSURNet


def h264_to_avi(path_264, path_avi):
    # 解压视频用
    # 我的C盘读写加上解压视频的时间都比磁盘直接读写快
    # 学姐要是有磁盘存并且磁盘读写速度足够快,可以解压完再训练
    if os.path.exists(path_avi):
        return

    stream = ffmpeg.input(path_264)
    stream = ffmpeg.output(stream, path_avi, vcodec='rawvideo')
    ffmpeg.run(stream)


def data_generator(patch_per_frame=64, patch_rate=0.4):
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
    trainVideos = list(range(44, 220))  # 在list里写上训练视频编号即可范围0-219
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


def training(model, device, patch_rate, epoch_nb=30):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.MSELoss()

    # minibatch 是一个压缩视频
    minibatch_mse = []
    minibatch_mae = []

    # 一个batch指一个源视频
    batch_mse_history = []
    batch_mae_history = []

    # 记录所有预测值与真值
    truth_sur = []
    predicted_sur = []

    # 实时绘图用
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    line1, = ax1.plot(batch_mae_history, label='train_MAE')
    line2, = ax1.plot(batch_mse_history, label='train_MSE')
    line3, = ax2.plot(truth_sur, label='truth_sur')
    line4, = ax2.plot(predicted_sur, label='predicted_sur')

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title("Training Label")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("truth/predicted SUR")
    ax2.legend()

    count = 0

    try:
        for epoch in range(epoch_nb):
            # 每个epoch需要创建新的数据生成器
            generator = data_generator(patch_per_frame=model.patch_per_frame, patch_rate=patch_rate)
            for X_train, y_train in generator:
                count += 1
                # 将数据移动到正确的设备（例如 GPU）
                X_train_gpu, y_train_gpu = tuple(x.to(device) for x in X_train), torch.tensor(y_train).float().to(device)
                optimizer.zero_grad()
                outputs = model(X_train_gpu)
                loss = criterion(outputs, y_train_gpu)
                loss.backward()
                optimizer.step()

                mse = loss.item()
                mae = abs(y_train-outputs.item())
                minibatch_mse.append(mse)
                minibatch_mae.append(mae)
                truth_sur.append(y_train)
                predicted_sur.append(outputs.item())
                print("******************** 当前压缩视频预测标签： ", outputs.item(), "真值： ", y_train_gpu.item())
                # loss = criterion(outputs, y_train_gpu)
                # loss.backward()
                # optimizer.step()
                if len(minibatch_mse) == 11:  # 完成了一个source video的SUR采样点训练
                    mean_mse = np.mean(minibatch_mse) if np.mean(minibatch_mse) < 10 else 10
                    mean_mae = np.mean(minibatch_mae) if np.mean(minibatch_mae) < 10 else 10
                    batch_mse_history.append(np.mean(mean_mse))
                    batch_mae_history.append(np.mean(mean_mae))
                    minibatch_mse = []
                    minibatch_mae = []
                    print("*******************************************************    MSE:  ", loss.item(), "   当前视频： ", count)

                # 实时绘图
                line3.set_ydata(truth_sur[-30:])
                line3.set_xdata(range(len(truth_sur[-30:])))
                line4.set_ydata(predicted_sur[-30:])
                line4.set_xdata(range(len(predicted_sur[-30:])))
                ax2.relim()
                ax2.autoscale_view()
                line1.set_ydata(batch_mae_history)
                line1.set_xdata(range(len(batch_mae_history)))
                line2.set_ydata(batch_mse_history)
                line2.set_xdata(range(len(batch_mse_history)))
                ax1.relim()
                ax1.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
            np.save('lossHistory/0111true_epoch'+str(epoch)+'.npy', truth_sur)
            np.save('lossHistory/0111pred_epoch' + str(epoch) + '.npy', predicted_sur)
            torch.save(model.state_dict(), 'weights/model_weights' + str(epoch) + '.pth')
    except Exception as err:
        print(f"训练过程中发生了错误：{err}", "     当前已训练视频：", count)
        traceback.print_exc()
    finally:
        plt.ioff()
        plt.show()
        np.save('lossHistory/batch_loss_final.npy', batch_mse_history)
        torch.save(model.state_dict(), 'weights/model_weights_final.pth')
        return batch_mse_history


def truncated_normal_(tensor, mean=0.0, std=1.0, trunc_std=2):
    """
    Fill the input Tensor with values drawn from a truncated normal distribution.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def initialize_weights(model):
    """
    Initialize the weights of the model with the truncated normal distribution.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            truncated_normal_(m.weight,mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.__version__)

    key_frame_nb = 12
    patch_per_frame = 64
    patch_rate = 0.4
    #  截断正态分布初始化
    model = STSURNet(key_frame_nb=key_frame_nb, patch_per_frame=patch_per_frame)
    initialize_weights(model)

    model.to(device)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total number of parameters: {total_params}")
    training(model, device, patch_rate=patch_rate)
