import matplotlib.pyplot as plt
import numpy as np
import torch
from dataGenerator import data_generator
from model.STSUR_Net import STSURNet
from scipy.special import erfc
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def testing(model, videoList = list(range(44))):
    model.eval()

    minibatch_mae = []
    minibatch_mse = []
    batch_mae_history = []
    batch_mse_history = []
    true_sur = []
    pred_sur = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2,1)
    line1, = ax1.plot(batch_mae_history,label='test_MAE')
    line2, = ax1.plot(batch_mse_history,label='test_MSE')
    line3, = ax2.plot(true_sur,label='true_sur')
    line4, = ax2.plot(pred_sur,label='pred_sur')
    ax1.set_title('Test Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.set_title('prediction label')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('label')
    ax2.legend()

    count = 0
    generator = data_generator(videoList=videoList,shuffle=False)
    for X_test,y_test in generator:
        count += 1
        X_test_gpu,y_test_gpu = tuple(x.to(device) for x in X_test),torch.tensor(y_test).to(device)
        outputs = model(X_test_gpu)
        outputs = torch.clamp(outputs, min=0, max=1)
        print('############################## 当前预测标签： ', outputs.item()," 真值： ", y_test, '  视频编号：',count)
        true_sur.append(y_test)
        pred_sur.append(outputs.item())

        mae = abs(outputs.item()-y_test)
        mse = (outputs.item()-y_test)**2
        minibatch_mae.append(mae)
        minibatch_mse.append(mse)

        if len(minibatch_mse) == 11:
            m_mae = np.mean(minibatch_mae)
            m_mse = np.mean(minibatch_mse)
            print('########################### mae：', m_mae, "  mse:", m_mse)
            batch_mae_history.append(m_mae)
            batch_mse_history.append(m_mse)
            minibatch_mae = []
            minibatch_mse = []

        # 实时绘图
        line3.set_ydata(true_sur[-30:])
        line3.set_xdata(range(len(true_sur[-30:])))
        line4.set_ydata(pred_sur[-30:])
        line4.set_xdata(range(len(pred_sur[-30:])))
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

    np.save('lossHistory/0113true.npy', true_sur)
    np.save('lossHistory/0113pred.npy', pred_sur)
    return true_sur,pred_sur


def normal_ccdf(x,mu,sigma):
    return 0.5*erfc((x-mu)/(sigma*np.sqrt(2)))


def sur_curve_fit(y):
    # print(y)
    top = [1,1,1,1,1,1]
    bottom = [0,0,0,0]
    temp = list(y)
    y = list(y)
    try:
        y.pop(0)
        y.pop(-1)
        x = [1,2,3,4,5,6, 7, 12, 17, 22, 27, 32, 37, 42, 47,48,49,50,51]
        y = top+y+bottom
        params, _ = curve_fit(normal_ccdf, x, y)
    except:
        x = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 51]
        y = temp
        params, _ = curve_fit(normal_ccdf, x, y)
    # x_vals = list(range(1, 52))
    x_vals = list(range(7, 48))
    y_vals = normal_ccdf(x_vals, *params)
    return list(y_vals)




if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # print(torch.__version__)
    #
    # key_frame_nb = 12
    # patch_per_frame = 64
    # patch_rate = 0.4
    #
    videoList = list(range(44))
    # model = STSURNet(key_frame_nb=key_frame_nb, patch_per_frame=patch_per_frame)
    # model.load_state_dict(torch.load('weights/1113model_weights49.pth'))
    # model.to(device)
    # true_sur,pred_sur = testing(model, videoList=videoList)

    true_sur = np.load('lossHistory/0113true.npy').reshape(44,11)
    pred_sur = np.load('lossHistory/0113pred.npy').reshape(44,11)
    true_list = []
    pred_list = []

    for i in range(len(videoList)):
        true_list += sur_curve_fit(true_sur[i])
        pred_list += sur_curve_fit(pred_sur[i])

    print(mean_squared_error(true_list,pred_list))
    print(mean_absolute_error(true_list, pred_list))
    print(r2_score(true_list, pred_list))
    np.save('lossHistory/0113predFited.npy',pred_list)



