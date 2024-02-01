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
from dataGenerator import data_generator


def training(model, device, patch_rate, epoch_nb=30):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
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
            generator = data_generator(videoList=list(range(44, 220)), patch_per_frame=model.patch_per_frame, patch_rate=patch_rate)
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
            np.save('lossHistory/0113true_epoch'+str(epoch)+'.npy', truth_sur)
            np.save('lossHistory/0113pred_epoch' + str(epoch) + '.npy', predicted_sur)
            torch.save(model.state_dict(), 'weights/1113model_weights' + str(epoch) + '.pth')
    except Exception as err:
        print(f"训练过程中发生了错误：{err}", "     当前已训练视频：", count)
        traceback.print_exc()
    finally:
        plt.ioff()
        plt.show()
        np.save('lossHistory/1113batch_loss_final.npy', batch_mse_history)
        torch.save(model.state_dict(), 'weights/1113model_weights_final.pth')
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
    # initialize_weights(model)
    model.load_state_dict(torch.load('weights/1113model_weights79.pth'))

    model.to(device)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total number of parameters: {total_params}")
    training(model, device, patch_rate=patch_rate,epoch_nb=200)
